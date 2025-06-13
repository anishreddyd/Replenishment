REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .episode_runner_with_base_stock import EpisodeRunnerWithBaseStock
REGISTRY["episode_with_base_stock"] = EpisodeRunnerWithBaseStock

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .whittle_cont_runner import WhittleContinuousRunner
REGISTRY["whittle_cont"] = WhittleContinuousRunner

from .parallel_runner_with_base_stock import ParallelRunnerWithBasestock
REGISTRY["parallel_with_base_stock"] = ParallelRunnerWithBasestock