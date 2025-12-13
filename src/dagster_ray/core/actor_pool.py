"""Prewarmed actor pool for executing Dagster runs without cold start overhead."""

from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import ray

logger = logging.getLogger(__name__)

# Type alias for Ray actor handles (untyped in Ray)
ActorHandle = Any

DAGSTER_WORKER_POOL_NAME = "DagsterWorkerPool"
DAGSTER_RAY_LAUNCHER_NAMESPACE = "dagster-ray-launcher"


class WorkerStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"
    INITIALIZING = "initializing"


@dataclass
class WorkerInfo:
    worker_id: str
    status: WorkerStatus
    current_run_id: str | None = None
    last_heartbeat: float | None = None


class DagsterRunWorker:
    """Actor that executes Dagster runs via subprocess.

    Each worker can execute one run at a time. The worker maintains its state
    and can be queried for status.
    """

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.status = WorkerStatus.IDLE
        self.current_run_id: str | None = None
        self.current_process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        logger.info(f"DagsterRunWorker {worker_id} initialized")

    def execute_run(self, run_id: str, entrypoint: str) -> dict[str, Any]:
        """Execute a Dagster run via the provided entrypoint command.

        Args:
            run_id: The Dagster run ID
            entrypoint: The shell command to execute (e.g., dagster api execute_run ...)

        Returns:
            Dict with execution result including return_code, stdout, stderr
        """
        with self._lock:
            if self.status == WorkerStatus.BUSY:
                return {
                    "success": False,
                    "error": f"Worker {self.worker_id} is busy with run {self.current_run_id}",
                }

            self.status = WorkerStatus.BUSY
            self.current_run_id = run_id

        try:
            logger.info(f"Worker {self.worker_id} starting run {run_id}")

            # Execute the entrypoint as a subprocess
            process = subprocess.Popen(
                entrypoint,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            with self._lock:
                self.current_process = process

            stdout, stderr = process.communicate()
            return_code = process.returncode

            logger.info(f"Worker {self.worker_id} completed run {run_id} with code {return_code}")

            return {
                "success": return_code == 0,
                "return_code": return_code,
                "stdout": stdout,
                "stderr": stderr,
                "worker_id": self.worker_id,
            }

        except Exception as e:
            logger.exception(f"Worker {self.worker_id} failed to execute run {run_id}")
            return {
                "success": False,
                "error": str(e),
                "worker_id": self.worker_id,
            }

        finally:
            with self._lock:
                self.status = WorkerStatus.IDLE
                self.current_run_id = None
                self.current_process = None

    def cancel_run(self) -> bool:
        """Cancel the currently running job if any."""
        with self._lock:
            if self.current_process is not None:
                logger.info(f"Worker {self.worker_id} canceling run {self.current_run_id}")
                self.current_process.terminate()
                try:
                    self.current_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.current_process.kill()
                return True
            return False

    def get_status(self) -> dict[str, Any]:
        """Get the current worker status."""
        with self._lock:
            return {
                "worker_id": self.worker_id,
                "status": self.status.value,
                "current_run_id": self.current_run_id,
            }

    def ping(self) -> str:
        """Health check."""
        return "pong"


class DagsterWorkerPool:
    """Actor that manages a pool of DagsterRunWorker actors.

    The pool maintains a set of prewarmed workers that can execute runs
    without cold start overhead. Workers are assigned to runs on a
    first-available basis.
    """

    def __init__(
        self,
        num_workers: int,
        worker_options: dict[str, Any] | None = None,
    ):
        self.num_workers = num_workers
        self.worker_options = worker_options or {}
        self.workers: list[ActorHandle] = []
        self.worker_ids: list[str] = []
        self.run_to_worker: dict[str, ActorHandle] = {}
        self.run_to_future: dict[str, Any] = {}  # ray.ObjectRef
        self._lock = threading.Lock()

        logger.info(f"Initializing DagsterWorkerPool with {num_workers} workers")
        self._create_workers()

    def _create_workers(self):
        """Create the worker actors."""
        import ray

        for i in range(self.num_workers):
            worker_id = f"worker-{i}"
            worker_cls = ray.remote(DagsterRunWorker)
            worker = worker_cls.options(  # type: ignore[attr-defined]
                name=f"dagster-run-worker-{i}",
                namespace=DAGSTER_RAY_LAUNCHER_NAMESPACE,
                **self.worker_options,
            ).remote(worker_id)
            # Ensure worker is ready
            ray.get(worker.ping.remote())  # type: ignore[union-attr]
            self.workers.append(worker)
            self.worker_ids.append(worker_id)
            logger.info(f"Created worker {worker_id}")

    def submit_run(self, run_id: str, entrypoint: str) -> dict[str, Any]:
        """Submit a run to an available worker.

        Args:
            run_id: The Dagster run ID
            entrypoint: The shell command to execute

        Returns:
            Dict with submission status and assigned worker
        """
        import ray

        with self._lock:
            # Find an available worker
            for worker in self.workers:
                status: dict[str, Any] = ray.get(worker.get_status.remote())  # type: ignore[union-attr]
                if status["status"] == WorkerStatus.IDLE.value:
                    # Submit the run asynchronously
                    future = worker.execute_run.remote(run_id, entrypoint)  # type: ignore[union-attr]
                    self.run_to_worker[run_id] = worker
                    self.run_to_future[run_id] = future
                    logger.info(f"Submitted run {run_id} to worker {status['worker_id']}")
                    return {
                        "success": True,
                        "worker_id": status["worker_id"],
                        "run_id": run_id,
                    }

            # No available workers
            logger.warning(f"No available workers for run {run_id}")
            return {
                "success": False,
                "error": "No available workers",
                "run_id": run_id,
            }

    def cancel_run(self, run_id: str) -> bool:
        """Cancel a running job."""
        import ray

        with self._lock:
            worker = self.run_to_worker.get(run_id)
            if worker is not None:
                result: bool = ray.get(worker.cancel_run.remote())  # type: ignore[union-attr]
                if result:
                    self.run_to_worker.pop(run_id, None)
                    self.run_to_future.pop(run_id, None)
                return result
            return False

    def get_run_status(self, run_id: str) -> dict[str, Any] | None:
        """Get the status of a specific run."""
        import ray

        with self._lock:
            future = self.run_to_future.get(run_id)
            worker = self.run_to_worker.get(run_id)

            if future is None:
                return None

            # Check if the future is ready
            ready, _ = ray.wait([future], timeout=0)
            if ready:
                result = ray.get(future)
                # Clean up
                self.run_to_worker.pop(run_id, None)
                self.run_to_future.pop(run_id, None)
                return {"completed": True, "result": result}
            else:
                if worker is not None:
                    worker_status = ray.get(worker.get_status.remote())  # type: ignore[union-attr]
                    return {"completed": False, "worker_status": worker_status}
                return {"completed": False, "worker_status": None}

    def get_pool_status(self) -> dict[str, Any]:
        """Get the status of all workers in the pool."""
        import ray

        worker_statuses: list[dict[str, Any]] = ray.get(
            [w.get_status.remote() for w in self.workers]  # type: ignore[union-attr]
        )
        idle_count = sum(1 for s in worker_statuses if s["status"] == WorkerStatus.IDLE.value)
        busy_count = sum(1 for s in worker_statuses if s["status"] == WorkerStatus.BUSY.value)

        return {
            "num_workers": self.num_workers,
            "idle_workers": idle_count,
            "busy_workers": busy_count,
            "workers": worker_statuses,
            "active_runs": list(self.run_to_worker.keys()),
        }

    def scale_workers(self, new_count: int) -> dict[str, Any]:
        """Scale the worker pool up or down.

        Args:
            new_count: The desired number of workers

        Returns:
            Dict with scaling result
        """
        import ray

        with self._lock:
            current_count = len(self.workers)

            if new_count > current_count:
                # Scale up
                for i in range(current_count, new_count):
                    worker_id = f"worker-{i}"
                    worker_cls = ray.remote(DagsterRunWorker)
                    worker = worker_cls.options(  # type: ignore[attr-defined]
                        name=f"dagster-run-worker-{i}",
                        namespace=DAGSTER_RAY_LAUNCHER_NAMESPACE,
                        **self.worker_options,
                    ).remote(worker_id)
                    ray.get(worker.ping.remote())  # type: ignore[union-attr]
                    self.workers.append(worker)
                    self.worker_ids.append(worker_id)
                    logger.info(f"Added worker {worker_id}")

            elif new_count < current_count:
                # Scale down - only remove idle workers
                workers_to_remove: list[ActorHandle] = []
                for worker in reversed(self.workers):
                    if len(self.workers) - len(workers_to_remove) <= new_count:
                        break
                    status: dict[str, Any] = ray.get(worker.get_status.remote())  # type: ignore[union-attr]
                    if status["status"] == WorkerStatus.IDLE.value:
                        workers_to_remove.append(worker)

                for worker in workers_to_remove:
                    idx = self.workers.index(worker)
                    self.workers.pop(idx)
                    self.worker_ids.pop(idx)
                    ray.kill(worker)
                    logger.info(f"Removed worker at index {idx}")

            self.num_workers = len(self.workers)

            return {
                "previous_count": current_count,
                "new_count": self.num_workers,
                "requested_count": new_count,
            }

    def shutdown(self):
        """Shutdown all workers in the pool."""
        import ray

        logger.info("Shutting down worker pool")
        with self._lock:
            for worker in self.workers:
                try:
                    ray.kill(worker)
                except Exception:
                    pass
            self.workers.clear()
            self.worker_ids.clear()
            self.run_to_worker.clear()
            self.run_to_future.clear()

    def ping(self) -> str:
        """Health check."""
        return "pong"

    @staticmethod
    def get_or_create(
        num_workers: int = 2,
        worker_options: dict[str, Any] | None = None,
    ) -> ActorHandle:
        """Get or create the worker pool actor.

        Args:
            num_workers: Number of workers to create
            worker_options: Ray actor options for workers (num_cpus, num_gpus, etc.)

        Returns:
            The worker pool actor handle
        """
        import ray

        pool_cls = ray.remote(DagsterWorkerPool)
        pool = pool_cls.options(  # type: ignore[attr-defined]
            name=DAGSTER_WORKER_POOL_NAME,
            namespace=DAGSTER_RAY_LAUNCHER_NAMESPACE,
            get_if_exists=True,
            lifetime="detached",
            max_concurrency=1000,
        ).remote(num_workers=num_workers, worker_options=worker_options)

        # Ensure pool is ready
        ray.get(pool.ping.remote())  # type: ignore[union-attr]

        return pool

    @staticmethod
    def get_existing() -> ActorHandle | None:
        """Get an existing worker pool actor if it exists.

        Returns:
            The worker pool actor handle or None if it doesn't exist
        """
        import ray

        try:
            pool = ray.get_actor(
                DAGSTER_WORKER_POOL_NAME,
                namespace=DAGSTER_RAY_LAUNCHER_NAMESPACE,
            )
            return pool
        except ValueError:
            return None

    @staticmethod
    def destroy_existing():
        """Destroy an existing worker pool if it exists."""
        import ray

        pool = DagsterWorkerPool.get_existing()
        if pool is not None:
            try:
                ray.get(pool.shutdown.remote())
            except Exception:
                pass
            try:
                ray.kill(pool)
            except Exception:
                pass

