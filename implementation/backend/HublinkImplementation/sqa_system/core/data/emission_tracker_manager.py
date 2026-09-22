import os
from pydantic import BaseModel

from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)

# Because of dependency issues between the GraphRAG package, the current codecarbon
# implementation is not working with it, which is why we supply it as an optional
# dependency. If codecarbon is not installed, we use a dummy implementation of the
# EmissionsTracker class.
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False

    class EmissionsTracker:
        """Dummy implementation of EmissionsTracker when CodeCarbon is not available."""

        def __init__(self):
            logger.warning(
                "CodeCarbon is not installed. Emissions tracking is disabled.")

        def start(self):
            """Not actually tracking, just a placeholder."""
            logger.debug(
                "Dummy EmissionsTracker.start() called. No tracking will occur.")

        def stop(self):
            """Not actually tracking, just a placeholder."""
            logger.debug(
                "Dummy EmissionsTracker.stop() called. Returning default data.")
            # Return a dummy object with the expected attributes.

            class DummyData:
                cpu_count = 0
                cpu_model = ""
                gpu_model = ""
                gpu_count = 0
                timestamp = ""
                duration = 0.0
                os = ""
                cpu_energy = 0.0
                gpu_energy = 0.0
                ram_energy = 0.0
                energy_consumed = 0.0
                emissions = 0.0
            # Mimic the structure expected by your EmissionTrackerManager.
            return DummyData()


class EmissionsTrackingData(BaseModel):
    """
    Data class for tracking emissions data.
    """
    cpu_count: int
    cpu_model: str
    gpu_model: str
    gpu_count: int
    timestamp: str
    tracking_duration: float
    os: str
    cpu_energy_consumption: float
    gpu_energy_consumption: float
    ram_energy_consumption: float
    total_energy_consumption: float
    emissions: float


class EmissionTrackerManager:
    """
    Manager class responsible for tracking emissions data with the CodeCarbon library.
    https://pypi.org/project/codecarbon/

    It is implemented as a singleton to ensure that only one instance of the
    emission tracker is running at any given time.
    """

    _instance = None
    _initialized = False
    tracker = None
    _disabled = False

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(EmissionTrackerManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # There currently is an issue regarding codecarbon locking when the
            # running application is abruptly terminated. This is a fix for this
            # issue as suggested here: https://github.com/mlco2/codecarbon/pull/701

            # if we have a linux system we can use /tmp/.codecarbon.lock
            # but if we have a windows system we need to lookup the appdata\Local\Temp\.codecarbon.lock
            if os.name == "nt":
                lock_file = os.path.join(
                    os.getenv("LOCALAPPDATA"), "Temp", ".codecarbon.lock")
            else:
                lock_file = "/tmp/.codecarbon.lock"
            if os.path.exists(lock_file):
                logger.debug("Removing CODECARBON lock file")
                os.remove(lock_file)
            self._initialize_tracker()
            self._initialized = True

    def disable(self):
        """
        Disables the emission tracker. This is useful in parallel runs
        as it is not working properly in this environment as stated here
        https://github.com/mlco2/codecarbon/issues/681
        """
        self._disabled = True
        self.tracker = None

    def enable(self):
        """
        Enables the emission tracker.
        """
        self._disabled = False
        self._initialize_tracker()

    def start(self):
        """Starts the emission tracker."""
        if not self._disabled:
            self.tracker.start()

    def stop_and_get_results(self) -> EmissionsTrackingData:
        """
        Stops the emission tracker and returns the final emissions data.
        The logic of stopping and returning results in one call comes 
        from the CodeCarbon library.
        
        Returns:
            EmissionsTrackingData: The emissions tracking object filled 
                with data about the tracking.
        """
        if self._disabled:
            logger.debug("Emission tracking is disabled.")
            return self._get_empty_result()

        stop_result = self.tracker.stop()
        try:
            if CODECARBON_AVAILABLE:
                emissions_data = getattr(
                    self.tracker, "final_emissions_data", None) or stop_result
            else:
                emissions_data = stop_result

            logger.debug(f"Stopping emission tracking: {emissions_data}")
            tracking_data = EmissionsTrackingData(
                cpu_count=emissions_data.cpu_count or 0,
                cpu_model=emissions_data.cpu_model or "",
                gpu_model=emissions_data.gpu_model or "",
                gpu_count=emissions_data.gpu_count or 0,
                timestamp=emissions_data.timestamp or "",
                tracking_duration=emissions_data.duration or 0.0,
                os=emissions_data.os or "",
                cpu_energy_consumption=emissions_data.cpu_energy or 0.0,
                gpu_energy_consumption=emissions_data.gpu_energy or 0.0,
                ram_energy_consumption=emissions_data.ram_energy or 0.0,
                total_energy_consumption=emissions_data.energy_consumed or 0.0,
                emissions=emissions_data.emissions or 0.0,
            )
        except Exception:
            logger.debug("Could not get results for emissions tracking.")
            tracking_data = self._get_empty_result()
        self._initialize_tracker()
        return tracking_data

    def _initialize_tracker(self):
        """
        Initializes the emission tracker if it is not already initialized.
        """
        self.tracker = EmissionsTracker(
            save_to_file=False,
            logging_logger=None,
            save_to_logger=False,
            log_level="error"
        )

    def _get_empty_result(self) -> EmissionsTrackingData:
        """
        Returns an empty result object.
        
        Returns:
            EmissionsTrackingData: An empty emissions tracking object.
        """
        return EmissionsTrackingData(
            cpu_count=0,
            cpu_model="",
            gpu_model="",
            gpu_count=0,
            timestamp="",
            tracking_duration=0.0,
            os="",
            cpu_energy_consumption=0.0,
            gpu_energy_consumption=0.0,
            ram_energy_consumption=0.0,
            total_energy_consumption=0.0,
            emissions=0.0,
        )
