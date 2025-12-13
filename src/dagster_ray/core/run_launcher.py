from __future__ import annotations

import functools
import logging
import sys
from typing import TYPE_CHECKING, Any, cast

import dagster as dg
from dagster._cli.api import ExecuteRunArgs  # type: ignore
from dagster._config.config_schema import UserConfigSchema
from dagster._core.events import EngineEventData
from dagster._core.launcher import LaunchRunContext, ResumeRunContext, RunLauncher
from dagster._core.launcher.base import CheckRunHealthResult, WorkerStatus

try:
    from dagster._core.remote_representation.origin import RemoteJobOrigin
except ImportError:
    # for new versions of dagster > 1.11.6
    from dagster._core.remote_origin import RemoteJobOrigin  # pyright: ignore[reportMissingImports]
from dagster._core.storage.dagster_run import DagsterRun, DagsterRunStatus
from dagster._grpc.types import ResumeRunArgs
from dagster._serdes import ConfigurableClass, ConfigurableClassData
from dagster._utils.error import serializable_error_info_from_exc_info
from packaging.version import Version
from pydantic import Field

from dagster_ray.configs import ActorPoolConfig, RayExecutionConfig, RayJobSubmissionClientConfig
from dagster_ray.core.actor_pool import DagsterWorkerPool
from dagster_ray.utils import resolve_env_vars_list

if TYPE_CHECKING:
    from ray.job_submission import JobSubmissionClient


def get_job_submission_id_from_run_id(run_id: str, resume_attempt_number=None):
    return f"dagster-run-{run_id}" + ("" if not resume_attempt_number else f"-{resume_attempt_number}")


class RayLauncherConfig(RayExecutionConfig, RayJobSubmissionClientConfig):
    env_vars: list[str] | None = Field(
        default=None,
        description="A list of environment variables to inject into the Job. Each can be of the form KEY=VALUE or just KEY (in which case the value will be pulled from the current process).",
    )
    actor_pool: ActorPoolConfig = Field(
        default_factory=ActorPoolConfig,
        description="Configuration for the prewarmed actor pool. When enabled, runs are executed by reusable actors instead of isolated jobs.",
    )


class RayRunLauncher(RunLauncher, ConfigurableClass):
    """RunLauncher that submits Dagster runs as isolated Ray jobs to a Ray cluster.

    Configuration can be provided via `dagster.yaml` and individual runs can override
    settings using the `dagster-ray/config` tag.

    The launcher supports two execution modes:
    1. **Job Submission** (default): Each run is submitted as an isolated Ray job
    2. **Actor Pool**: Runs are executed by prewarmed, reusable actors for faster startup

    Example:
        Configure via `dagster.yaml`
        ```yaml
        run_launcher:
          module: dagster_ray
          class: RayRunLauncher
          config:
            address: "ray://head-node:10001"
            num_cpus: 2
            num_gpus: 0
        ```

    Example:
        Enable prewarmed actor pool for faster run startup
        ```yaml
        run_launcher:
          module: dagster_ray
          class: RayRunLauncher
          config:
            address: "ray://head-node:10001"
            actor_pool:
              enabled: true
              num_workers: 4
              worker_num_cpus: 2
              worker_num_gpus: 0
              auto_scale: true
              max_workers: 10
        ```

    Example:
        Override settings per job
        ```python
        import dagster as dg

        @dg.job(
            tags={
                "dagster-ray/config": {
                    "num_cpus": 16,
                    "num_gpus": 1,
                    "runtime_env": {"pip": {"packages": ["torch"]}},
                }
            }
        )
        def my_job():
            return my_op()
        ```

    Example:
        Programmatically interact with the actor pool
        ```python
        from dagster_ray import RayRunLauncher

        launcher = instance.run_launcher  # Get the run launcher from your Dagster instance

        # Prewarm the pool before runs
        status = launcher.prewarm_pool()
        print(f"Pool has {status['idle_workers']} idle workers")

        # Scale the pool dynamically
        launcher.scale_pool(num_workers=8)

        # Get pool status
        status = launcher.get_pool_status()

        # Shutdown the pool when done
        launcher.shutdown_pool()
        ```
    """

    def __init__(
        self,
        address: str,
        metadata: dict[str, Any] | None = None,
        headers: dict[str, Any] | None = None,
        cookies: dict[str, Any] | None = None,
        env_vars: list[str] | None = None,
        runtime_env: dict[str, Any] | None = None,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        memory: int | None = None,
        resources: dict[str, float] | None = None,
        actor_pool: dict[str, Any] | None = None,
        inst_data: ConfigurableClassData | None = None,
    ):
        self._inst_data = dg._check.opt_inst_param(inst_data, "inst_data", ConfigurableClassData)

        self.address = address
        self.metadata = metadata
        self.headers = headers
        self.cookies = cookies
        self.env_vars = env_vars
        self.runtime_env = runtime_env
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.memory = memory
        self.resources = resources

        # Actor pool configuration
        self.actor_pool_config = ActorPoolConfig(**(actor_pool or {}))
        self._worker_pool = None

        super().__init__()

    @functools.cached_property
    def client(self) -> JobSubmissionClient:
        '''
        Cache the client to avoid creating multiple connections.
        When using ray:// addresses, JobSubmissionClient internally calls ray.init()
        which fails if already connected.
        '''
        from ray.job_submission import JobSubmissionClient

        return JobSubmissionClient(self.address, metadata=self.metadata, headers=self.headers, cookies=self.cookies)

    @property
    def worker_pool(self):
        """Get or create the worker pool if actor pool is enabled."""
        if not self.actor_pool_config.enabled:
            return None

        if self._worker_pool is None:
            import ray

            # Initialize Ray if not already connected
            if not ray.is_initialized():
                ray.init(address=self.address, ignore_reinit_error=True)

            worker_options = {}
            if self.actor_pool_config.worker_num_cpus is not None:
                worker_options["num_cpus"] = self.actor_pool_config.worker_num_cpus
            if self.actor_pool_config.worker_num_gpus is not None:
                worker_options["num_gpus"] = self.actor_pool_config.worker_num_gpus
            if self.actor_pool_config.worker_memory is not None:
                worker_options["memory"] = self.actor_pool_config.worker_memory
            if self.actor_pool_config.worker_resources is not None:
                worker_options["resources"] = self.actor_pool_config.worker_resources
            if self.actor_pool_config.worker_runtime_env is not None:
                worker_options["runtime_env"] = self.actor_pool_config.worker_runtime_env

            self._worker_pool = DagsterWorkerPool.get_or_create(
                num_workers=self.actor_pool_config.num_workers,
                worker_options=worker_options if worker_options else None,
            )

        return self._worker_pool

    def prewarm_pool(self) -> dict[str, Any]:
        """Prewarm the actor pool. Call this before launching runs to ensure workers are ready.

        Returns:
            Dict with pool status information
        """
        import ray

        if not self.actor_pool_config.enabled:
            return {"error": "Actor pool is not enabled"}

        pool = self.worker_pool
        if pool is None:
            return {"error": "Failed to create worker pool"}
        return ray.get(pool.get_pool_status.remote())  # type: ignore[union-attr]

    def get_pool_status(self) -> dict[str, Any] | None:
        """Get the current status of the actor pool.

        Returns:
            Dict with pool status or None if pool is not enabled
        """
        import ray

        if not self.actor_pool_config.enabled or self._worker_pool is None:
            return None

        return ray.get(self._worker_pool.get_pool_status.remote())

    def scale_pool(self, num_workers: int) -> dict[str, Any]:
        """Scale the actor pool to the specified number of workers.

        Args:
            num_workers: The desired number of workers

        Returns:
            Dict with scaling result
        """
        import ray

        if not self.actor_pool_config.enabled:
            return {"error": "Actor pool is not enabled"}

        pool = self.worker_pool
        if pool is None:
            return {"error": "Failed to create worker pool"}
        return ray.get(pool.scale_workers.remote(num_workers))  # type: ignore[union-attr]

    def shutdown_pool(self):
        """Shutdown the actor pool and all workers."""
        import ray

        if self._worker_pool is not None:
            try:
                ray.get(self._worker_pool.shutdown.remote())
                ray.kill(self._worker_pool)
            except Exception:
                pass
            self._worker_pool = None

    @property
    def inst_data(self) -> ConfigurableClassData | None:
        return self._inst_data

    @classmethod
    def config_type(cls) -> UserConfigSchema:
        return {"ray": RayLauncherConfig.to_config_schema().as_field()}

    @classmethod
    def from_config_value(cls, inst_data, config_value):
        return cls(inst_data=inst_data, **config_value["ray"])

    @property
    def supports_resume_run(self):
        return True

    @property
    def supports_check_run_worker_health(self):
        return True

    @property
    def supports_run_worker_crash_recovery(self):
        return True

    def launch_run(self, context: LaunchRunContext) -> None:
        run = context.dagster_run
        submission_id = get_job_submission_id_from_run_id(run.run_id)
        job_origin = dg._check.not_none(run.job_code_origin)

        args = list(
            ExecuteRunArgs(
                job_origin=job_origin,
                run_id=run.run_id,
                instance_ref=self._instance.get_ref(),
                set_exit_code_on_failure=True,
            ).get_command_args()
        )

        # wrap the json in quotes to prevent errors with shell commands
        args[-1] = "'" + args[-1] + "'"

        entrypoint = " ".join(args)

        # Use actor pool if enabled
        if self.actor_pool_config.enabled:
            self._launch_via_actor_pool(run, entrypoint)
        else:
            self._launch_ray_job(submission_id, entrypoint, run)

    def _launch_via_actor_pool(self, run: DagsterRun, entrypoint: str) -> None:
        """Launch a run using the prewarmed actor pool."""
        import ray
        from typing import Any

        self._instance.report_engine_event(
            "Submitting run to actor pool",
            run,
            EngineEventData(
                {
                    "Run ID": run.run_id,
                    "Pool Workers": str(self.actor_pool_config.num_workers),
                }
            ),
            cls=self.__class__,
        )

        pool = self.worker_pool

        # Check pool status and auto-scale if needed
        if self.actor_pool_config.auto_scale:
            pool_status: dict[str, Any] = ray.get(pool.get_pool_status.remote())  # type: ignore[union-attr]
            if pool_status["idle_workers"] == 0 and pool_status["num_workers"] < self.actor_pool_config.max_workers:
                new_count = min(pool_status["num_workers"] + 1, self.actor_pool_config.max_workers)
                ray.get(pool.scale_workers.remote(new_count))  # type: ignore[union-attr]
                self._instance.report_engine_event(
                    f"Auto-scaled actor pool to {new_count} workers",
                    run,
                    cls=self.__class__,
                )

        # Submit the run to the pool
        result: dict[str, Any] = ray.get(pool.submit_run.remote(run.run_id, entrypoint))  # type: ignore[union-attr]

        if result["success"]:
            self._instance.report_engine_event(
                "Run submitted to actor pool",
                run,
                EngineEventData(
                    {
                        "Worker ID": result["worker_id"],
                        "Run ID": run.run_id,
                    }
                ),
                cls=self.__class__,
            )
        else:
            if self.actor_pool_config.queue_runs:
                # Queue the run - it will be picked up when a worker becomes available
                self._instance.report_engine_event(
                    "No available workers, run queued",
                    run,
                    EngineEventData({"Error": result.get("error", "Unknown")}),
                    cls=self.__class__,
                )
                # Retry submission in a background task
                self._queue_run_for_pool(run, entrypoint)
            else:
                # Fall back to job submission
                self._instance.report_engine_event(
                    "No available workers, falling back to job submission",
                    run,
                    cls=self.__class__,
                )
                submission_id = get_job_submission_id_from_run_id(run.run_id)
                self._launch_ray_job(submission_id, entrypoint, run)

    def _queue_run_for_pool(self, run: DagsterRun, entrypoint: str) -> None:
        """Queue a run for execution when a worker becomes available."""
        import ray
        import time
        from typing import Any

        @ray.remote  # type: ignore[misc]
        def wait_and_submit(pool_handle: Any, run_id: str, entrypoint: str, max_wait: int = 300) -> dict[str, Any]:
            """Wait for an available worker and submit the run."""
            start_time = time.time()
            while time.time() - start_time < max_wait:
                result: dict[str, Any] = ray.get(pool_handle.submit_run.remote(run_id, entrypoint))
                if result["success"]:
                    return result
                time.sleep(1)
            return {"success": False, "error": "Timeout waiting for available worker"}

        # Submit the queuing task asynchronously
        wait_and_submit.remote(self.worker_pool, run.run_id, entrypoint)  # type: ignore[attr-defined]

    def _launch_ray_job(self, submission_id: str, entrypoint: str, run: DagsterRun):
        # note: entrypoint is a shell command
        job_origin = dg._check.not_none(run.job_code_origin)

        labels = {
            "dagster/job": job_origin.job_name,
            "dagster/run-id": run.run_id,
        }

        if Version(dg.__version__) >= Version("1.8.12"):
            remote_job_origin = run.remote_job_origin  # type: ignore
        else:
            remote_job_origin = run.external_job_origin  # type: ignore

        remote_job_origin = cast(RemoteJobOrigin | None, remote_job_origin)

        if remote_job_origin:
            labels["dagster/code-location"] = remote_job_origin.repository_origin.code_location_origin.location_name

        cfg_from_tags = RayLauncherConfig.from_tags(run.tags)

        env_vars = cfg_from_tags.env_vars or self.env_vars or []
        # note! ray modifies the user-provided runtime_env, so we copy it
        runtime_env = (cfg_from_tags.runtime_env or self.runtime_env or {}).copy()
        num_cpus = cfg_from_tags.num_cpus or self.num_cpus
        num_gpus = cfg_from_tags.num_gpus or self.num_gpus
        memory = cfg_from_tags.memory or self.memory
        resources = cfg_from_tags.resources or self.resources

        runtime_env["env_vars"] = runtime_env.get("env_vars", {})
        runtime_env["env_vars"].update(resolve_env_vars_list(env_vars))
        runtime_env["env_vars"].update(
            {
                "DAGSTER_RUN_JOB_NAME": job_origin.job_name,
            }
        )

        self._instance.report_engine_event(
            "Creating Ray run job",
            run,
            EngineEventData(
                {
                    "Ray Job Submission ID": submission_id,
                    "Run ID": run.run_id,
                }
            ),
            cls=self.__class__,
        )

        self.client.submit_job(
            submission_id=submission_id,
            entrypoint=entrypoint,
            runtime_env=runtime_env,
            entrypoint_num_cpus=num_cpus,
            entrypoint_num_gpus=num_gpus,
            entrypoint_memory=memory,
            entrypoint_resources=resources,
            metadata=labels,
        )

        self._instance.report_engine_event(
            "Ray run job created",
            run,
            cls=self.__class__,
        )

    def resume_run(self, context: ResumeRunContext) -> None:
        run = context.dagster_run
        submission_id = get_job_submission_id_from_run_id(
            run.run_id, resume_attempt_number=context.resume_attempt_number
        )
        job_origin = dg._check.not_none(run.job_code_origin)

        args = list(
            ResumeRunArgs(
                job_origin=job_origin,
                run_id=run.run_id,
                instance_ref=self._instance.get_ref(),
                set_exit_code_on_failure=True,
            ).get_command_args()
        )

        # wrap the json in quotes to prevent erros with shell commands
        args[-1] = "'" + args[-1] + "'"

        entrypoint = " ".join(args)

        # Use actor pool if enabled
        if self.actor_pool_config.enabled:
            self._launch_via_actor_pool(run, entrypoint)
        else:
            self._launch_ray_job(submission_id, entrypoint, run)

    def terminate(self, run_id: str) -> bool:
        dg._check.str_param(run_id, "run_id")
        run = self._instance.get_run_by_id(run_id)

        if not run or run.is_finished:
            return False

        self._instance.report_run_canceling(run)

        # Try actor pool termination first if enabled
        if self.actor_pool_config.enabled and self._worker_pool is not None:
            try:
                import ray

                result = ray.get(self._worker_pool.cancel_run.remote(run_id))
                if result:
                    self._instance.report_engine_event(
                        message="Run was terminated successfully via actor pool.",
                        dagster_run=run,
                        cls=self.__class__,
                    )
                    return True
            except Exception:
                pass  # Fall through to job submission termination

        submission_id = get_job_submission_id_from_run_id(
            run.run_id, resume_attempt_number=self._instance.count_resume_run_attempts(run.run_id)
        )

        try:
            termination_result = self.client.stop_job(submission_id)
            if termination_result:
                self._instance.report_engine_event(
                    message="Run was terminated successfully.",
                    dagster_run=run,
                    cls=self.__class__,
                )
            else:
                self._instance.report_engine_event(
                    message="Run was terminated succesfully (the Ray job was already terminated).",
                    dagster_run=run,
                    cls=self.__class__,
                )
            return termination_result
        except RuntimeError:
            self._instance.report_engine_event(
                message="Run was not terminated successfully; encountered error in stop_job",
                dagster_run=run,
                engine_event_data=EngineEventData.engine_error(serializable_error_info_from_exc_info(sys.exc_info())),
                cls=self.__class__,
            )
            return False

    def get_run_worker_debug_info(self, run: DagsterRun, include_container_logs: bool | None = True) -> str | None:
        try:
            job_details = [j for j in self.client.list_jobs() if (j.metadata or {}).get("dagster/run-id") == run.run_id]

            return "---\n---".join(str(j) for j in job_details)

        except RuntimeError:
            logging.exception("Error trying to get debug information for failed Ray jobs")

    def check_run_worker_health(self, run: DagsterRun):
        from ray.job_submission import JobStatus
        from typing import Any

        # Check actor pool first if enabled
        if self.actor_pool_config.enabled and self._worker_pool is not None:
            try:
                import ray

                run_status: dict[str, Any] | None = ray.get(
                    self._worker_pool.get_run_status.remote(run.run_id)  # type: ignore[union-attr]
                )
                if run_status is not None:
                    if run_status["completed"]:
                        result: dict[str, Any] = run_status["result"]
                        if result.get("success"):
                            return CheckRunHealthResult(WorkerStatus.SUCCESS)
                        else:
                            return CheckRunHealthResult(
                                WorkerStatus.FAILED,
                                f"Actor pool run failed: {result.get('error', result.get('stderr', 'Unknown error'))}",
                            )
                    else:
                        return CheckRunHealthResult(WorkerStatus.RUNNING)
            except Exception:
                pass  # Fall through to job submission check

        if self.supports_run_worker_crash_recovery:
            resume_attempt_number = self._instance.count_resume_run_attempts(run.run_id)
        else:
            resume_attempt_number = None

        submission_id = get_job_submission_id_from_run_id(run.run_id, resume_attempt_number=resume_attempt_number)

        try:
            status = self.client.get_job_status(submission_id)
        except RuntimeError:
            return CheckRunHealthResult(
                WorkerStatus.UNKNOWN, str(serializable_error_info_from_exc_info(sys.exc_info()))
            )

        # If the run is in a non-terminal (and non-STARTING) state but the ray job is not active,
        # something went wrong
        if run.status in (DagsterRunStatus.STARTED, DagsterRunStatus.CANCELING) and status.is_terminal():
            return CheckRunHealthResult(
                WorkerStatus.FAILED, f"Run has not completed but Ray job has is in status: {status}"
            )

        elif status == JobStatus.FAILED:
            job_details = self.client.get_job_info(submission_id)
            return CheckRunHealthResult(WorkerStatus.FAILED, f"Ray job failed. Message: {job_details.message}")
        elif status == JobStatus.STOPPED:
            job_details = self.client.get_job_info(submission_id)
            return CheckRunHealthResult(
                WorkerStatus.FAILED, f"Ray job has been stopped externally. Message: {job_details.message}"
            )
        elif status == JobStatus.SUCCEEDED:
            return CheckRunHealthResult(WorkerStatus.SUCCESS)
        elif status in {JobStatus.RUNNING, JobStatus.PENDING}:
            return CheckRunHealthResult(WorkerStatus.RUNNING)

        # safe return in case more statuses are introduced later
        return CheckRunHealthResult(WorkerStatus.RUNNING)
