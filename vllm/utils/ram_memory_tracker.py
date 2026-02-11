# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RAM memory tracking utility for vLLM startup process."""

import gc
import os
from typing import Optional

import psutil

from vllm.logger import init_logger

logger = init_logger(__name__)


class RAMMemoryTracker:
    """Track RAM memory usage during vLLM startup.
    
    This tracker monitors RSS (Resident Set Size) memory, which represents
    the actual physical memory used by the process. It handles dynamic memory
    changes such as:
    - Temporary allocations (e.g., model weights loaded to CPU then moved to GPU)
    - Memory reuse (freed memory being reused by other components)
    - Garbage collection effects
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.process = psutil.Process(os.getpid())
        self.baseline_memory_mb: float = 0.0
        self.last_memory_mb: float = 0.0
        # Track peak memory to detect memory reuse scenarios
        self.peak_memory_mb: float = 0.0
        # Track memory history for better analysis
        self.memory_history: list[tuple[str, float, float]] = []  # (name, delta, total)
        self._initialize_baseline()

    def _initialize_baseline(self) -> None:
        """Initialize baseline memory before any vLLM allocations."""
        if self.enabled:
            gc.collect()
            self.baseline_memory_mb = self._get_current_memory_mb()
            logger.info(
                f"内存跟踪基线: {self.baseline_memory_mb:.2f} M RAM内存（第二次改动）"
            )
            self.last_memory_mb = self.baseline_memory_mb
            self.peak_memory_mb = self.baseline_memory_mb

    def _get_current_memory_mb(self) -> float:
        """Get current process RSS memory in MB.
        
        RSS (Resident Set Size) is the actual physical memory used by the process.
        Note: When memory is freed and reused, RSS may not decrease immediately
        if the freed memory is reused by other allocations.
        """
        try:
            # RSS (Resident Set Size) is the actual physical memory used
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0

    def log_memory_usage(
        self, 
        component_name: str, 
        force_gc: bool = True,
        note: str | None = None
    ) -> None:
        """
        Log memory usage for a specific component.

        Args:
            component_name: Name of the component being tracked
            force_gc: Whether to force garbage collection before measuring
            note: Optional note about memory change (e.g., "释放", "转移到GPU")
        """
        if not self.enabled:
            return

        if force_gc:
            gc.collect()

        current_memory_mb = self._get_current_memory_mb()
        memory_delta_mb = current_memory_mb - self.last_memory_mb
        total_memory_mb = current_memory_mb - self.baseline_memory_mb

        # Update peak memory
        if current_memory_mb > self.peak_memory_mb:
            self.peak_memory_mb = current_memory_mb

        # Build log message
        # 同时显示增量、累计增量和绝对RSS值，便于与top命令对比
        if note:
            log_msg = (
                f"{component_name} {note} 占用 %.2f M RAM内存 "
                f"(累计增量: %.2f M, 绝对RSS: %.2f M)（第二次改动）"
            )
        else:
            log_msg = (
                f"{component_name} 占用 %.2f M RAM内存 "
                f"(累计增量: %.2f M, 绝对RSS: %.2f M)（第二次改动）"
            )

        # Log memory change
        # We log all changes, even small ones, to track memory dynamics
        logger.info(log_msg, memory_delta_mb, total_memory_mb, current_memory_mb)

        # Store in history for analysis
        self.memory_history.append((component_name, memory_delta_mb, total_memory_mb))

        self.last_memory_mb = current_memory_mb

    def log_memory_release(
        self, 
        component_name: str, 
        expected_release_mb: float | None = None,
        force_gc: bool = True
    ) -> None:
        """
        Explicitly log memory release (e.g., when weights move from CPU to GPU).
        
        This helps track memory that is freed and may be reused by other components.
        
        Args:
            component_name: Name of the component releasing memory
            expected_release_mb: Expected amount of memory to be released (for validation)
            force_gc: Whether to force garbage collection before measuring
        """
        if not self.enabled:
            return

        if force_gc:
            gc.collect()

        current_memory_mb = self._get_current_memory_mb()
        memory_delta_mb = current_memory_mb - self.last_memory_mb
        total_memory_mb = current_memory_mb - self.baseline_memory_mb

        # If memory decreased, log the release
        if memory_delta_mb < 0:
            release_msg = f"{component_name} 释放 %.2f M RAM内存 (累计: %.2f M)（第二次改动）"
            if expected_release_mb and abs(memory_delta_mb) < expected_release_mb * 0.5:
                # Memory might be reused, add a note
                release_msg += f" [注意: 可能部分内存已被其他模块重用]"
            logger.info(release_msg, abs(memory_delta_mb), total_memory_mb)
        else:
            # Memory didn't decrease - might be reused immediately
            logger.info(
                "%s 尝试释放内存，但RSS未减少 (变化: %.2f M, 累计: %.2f M) "
                "[可能原因: 内存被其他模块立即重用，或Python未立即释放]（第二次改动）",
                component_name,
                memory_delta_mb,
                total_memory_mb,
            )

        self.memory_history.append((f"{component_name}(释放)", memory_delta_mb, total_memory_mb))
        self.last_memory_mb = current_memory_mb

    def get_total_memory_usage_mb(self) -> float:
        """Get total memory usage since baseline."""
        if not self.enabled:
            return 0.0
        current_memory_mb = self._get_current_memory_mb()
        return current_memory_mb - self.baseline_memory_mb


# Global tracker instance
_global_tracker: Optional[RAMMemoryTracker] = None


def get_ram_memory_tracker() -> RAMMemoryTracker:
    """Get or create the global RAM memory tracker."""
    global _global_tracker
    if _global_tracker is None:
        # Check if tracking is enabled via environment variable
        enabled = os.environ.get("VLLM_ENABLE_RAM_TRACKING", "1") == "1"
        _global_tracker = RAMMemoryTracker(enabled=enabled)
    return _global_tracker


def log_ram_memory(
    component_name: str, 
    force_gc: bool = True,
    note: str | None = None
) -> None:
    """
    Convenience function to log RAM memory usage.

    Args:
        component_name: Name of the component being tracked
        force_gc: Whether to force garbage collection before measuring
        note: Optional note about memory change (e.g., "释放", "转移到GPU")
    """
    tracker = get_ram_memory_tracker()
    tracker.log_memory_usage(component_name, force_gc=force_gc, note=note)


def log_ram_memory_from_cpp(component_name: str, size_bytes: int) -> None:
    """
    Log RAM memory usage from C++ code.
    This function is designed to be called from C++ extensions.

    Args:
        component_name: Name of the component allocating memory
        size_bytes: Size of memory allocated in bytes
    """
    tracker = get_ram_memory_tracker()
    if not tracker.enabled:
        return
    
    import gc
    gc.collect()
    
    current_memory_mb = tracker._get_current_memory_mb()
    memory_delta_mb = current_memory_mb - tracker.last_memory_mb
    total_memory_mb = current_memory_mb - tracker.baseline_memory_mb
    
    # Update peak memory
    if current_memory_mb > tracker.peak_memory_mb:
        tracker.peak_memory_mb = current_memory_mb
    
    # Calculate expected memory increase from the allocation
    expected_mb = size_bytes / (1024.0 * 1024.0)
    
    # Log with note about C++ allocation
    log_msg = f"{component_name} 占用 {memory_delta_mb:.2f} M RAM内存 (累计: {total_memory_mb:.2f} M) [C++分配: {expected_mb:.2f} M]（第二次改动）"
    logger.info(log_msg)
    
    tracker.memory_history.append((component_name, memory_delta_mb, total_memory_mb))
    tracker.last_memory_mb = current_memory_mb


def log_ram_memory_release(
    component_name: str,
    expected_release_mb: float | None = None,
    force_gc: bool = True
) -> None:
    """
    Convenience function to log RAM memory release.

    Use this when you know memory should be released (e.g., weights moved to GPU).

    Args:
        component_name: Name of the component releasing memory
        expected_release_mb: Expected amount of memory to be released
        force_gc: Whether to force garbage collection before measuring
    """
    tracker = get_ram_memory_tracker()
    tracker.log_memory_release(component_name, expected_release_mb, force_gc)
