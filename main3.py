# milestone3_sensors.py
"""
Milestone 3: Sensor Measurement Modeling for Ceres Docking Operations
Complete implementation of camera and range sensor models with measurement analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

class CameraModel:
    """
    Pinhole camera model based on coordinate system framework from discussion slides
    Implements: [Image coordinates] = [Intrinsic] * [Extrinsic] * [World coordinates]
    """
    
    def __init__(self):
        # Camera intrinsic parameters
        self.focal_length = 0.05      # 50mm principal distance (slide 9)
        self.pixel_size = 1e-5        # 10 micron pixels
        self.image_width = 1024       # pixels
        self.image_height = 1024      # pixels
        self.principal_point = (self.image_width/2, self.image_height/2)
        
        # Camera field of view (45 degrees based on slide 9 concept)
        self.fov_x = np.radians(45)
        self.fov_y = np.radians(45)
        
        # Measurement noise (slide 11: Gaussian noise)
        self.pixel_noise_std = 0.5    # pixels measurement noise
        
        # Camera extrinsic parameters (simplified - aligned with body frame)
        self.R_camera_to_body = np.eye(3)  # Rotation matrix
        self.t_camera_to_body = np.zeros(3) # Translation vector
    
    def world_to_camera_frame(self, world_position, spacecraft_attitude=None):
        """
        Transform ISS position from world (CW) frame to camera frame
        Simplified: assumes camera aligned with body frame and body aligned with world
        """
        if spacecraft_attitude is None:
            spacecraft_attitude = np.eye(3)
            
        # Transform to camera frame: p_camera = R_camera_to_body @ R_body_to_world @ p_world + t
        camera_position = self.R_camera_to_body @ spacecraft_attitude.T @ world_position + self.t_camera_to_body
        return camera_position
    
    def project_to_image(self, camera_position):
        """
        Project 3D camera coordinates to 2D image coordinates using pinhole model
        Returns pixel coordinates or None if outside field of view
        """
        x_cam, y_cam, z_cam = camera_position
        
        # Check if ISS is in front of camera (slide 9 viewing direction)
        if z_cam <= 0:
            return None
            
        # Pinhole projection equations
        x_image = self.focal_length * x_cam / z_cam
        y_image = self.focal_length * y_cam / z_cam
        
        # Convert to pixel coordinates
        x_pixel = x_image / self.pixel_size + self.principal_point[0]
        y_pixel = y_image / self.pixel_size + self.principal_point[1]
        
        # Check if within image bounds and field of view
        if (0 <= x_pixel < self.image_width and 
            0 <= y_pixel < self.image_height and
            abs(x_image) <= self.focal_length * np.tan(self.fov_x/2) and
            abs(y_image) <= self.focal_length * np.tan(self.fov_y/2)):
            
            # Add Gaussian noise (slide 11)
            x_pixel += np.random.normal(0, self.pixel_noise_std)
            y_pixel += np.random.normal(0, self.pixel_noise_std)
            
            return np.array([x_pixel, y_pixel])
        else:
            return None  # Outside field of view
    
    def pixels_to_angles(self, pixel_coords):
        """
        Convert pixel coordinates to azimuth and elevation angles
        """
        x_pixel, y_pixel = pixel_coords
        
        # Convert back to image coordinates
        x_image = (x_pixel - self.principal_point[0]) * self.pixel_size
        y_image = (y_pixel - self.principal_point[1]) * self.pixel_size
        
        # Convert to angles
        azimuth = np.arctan2(x_image, self.focal_length)
        elevation = np.arctan2(y_image, self.focal_length)
        
        return np.array([azimuth, elevation])


class RangeSensor:
    """
    Range sensor model for measuring distance to ISS
    """
    
    def __init__(self):
        # Sensor characteristics
        self.range_noise_std = 0.1    # 10 cm measurement noise (slide 11)
        self.max_range = 10000        # 10 km maximum detection range
        self.min_range = 1            # 1 m minimum detection range
    
    def measure_range(self, world_position):
        """
        Measure range to ISS with Gaussian noise
        Returns measured range in meters or None if outside operational range
        """
        true_range = np.linalg.norm(world_position)
        
        # Check operational range
        if true_range > self.max_range or true_range < self.min_range:
            return None
            
        # Add Gaussian noise: Measured = True + Noise (slide 11)
        measured_range = true_range + np.random.normal(0, self.range_noise_std)
        return max(measured_range, 0)  # Ensure non-negative


class SensorSuite:
    """
    Integrated sensor suite combining camera and range sensors
    """
    
    def __init__(self):
        self.camera = CameraModel()
        self.range_sensor = RangeSensor()
        
    def get_measurements(self, true_position, time, spacecraft_attitude=None):
        """
        Get all available sensor measurements at current time
        Returns dictionary with measurements or None if no measurements available
        """
        measurements = {}
        measurements['time'] = time
        measurements['true_position'] = true_position.copy()
        
        # Range measurement (primary sensor)
        range_meas = self.range_sensor.measure_range(true_position)
        if range_meas is not None:
            measurements['range'] = range_meas
        
        # Camera measurements
        camera_pos = self.camera.world_to_camera_frame(true_position, spacecraft_attitude)
        pixel_coords = self.camera.project_to_image(camera_pos)
        
        if pixel_coords is not None:
            measurements['pixels'] = pixel_coords
            measurements['angles'] = self.camera.pixels_to_angles(pixel_coords)
        
        # Return None if no measurements available
        if len(measurements) <= 2:  # Only time and true_position
            return None
            
        return measurements
    
    def measurement_availability(self, true_position, spacecraft_attitude=None):
        """
        Check which measurements are available given current geometry
        Returns dictionary with availability flags
        """
        availability = {}
        
        # Range sensor availability
        range_val = np.linalg.norm(true_position)
        availability['range_available'] = (
            self.range_sensor.min_range <= range_val <= self.range_sensor.max_range
        )
        
        # Camera availability
        camera_pos = self.camera.world_to_camera_frame(true_position, spacecraft_attitude)
        pixel_coords = self.camera.project_to_image(camera_pos)
        availability['camera_available'] = (pixel_coords is not None)
        
        return availability


class NavigationFilter:
    """
    Conceptual Extended Kalman Filter for state estimation
    Demonstrates how measurements would be used to correct propagated state
    """
    
    def __init__(self, initial_state, initial_covariance, n=0.00113):
        """
        Initialize filter with initial state estimate and uncertainty
        n: orbital rate (rad/s)
        """
        self.state = initial_state.copy()  # [x, y, z, vx, vy, vz]
        self.covariance = initial_covariance.copy()
        self.n = n
        
        # Process noise covariance (tuned for CW dynamics)
        self.Q = np.diag([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8])
        
        # Measurement noise covariances
        self.R_range = np.array([[0.01]])  # Range noise (0.1m std)^2
        self.R_angles = np.diag([1e-6, 1e-6])  # Angle noise
        
        # CW dynamics matrices
        self.A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3*self.n**2, 0, 0, 0, 2*self.n, 0],
            [0, 0, 0, -2*self.n, 0, 0],
            [0, 0, -self.n**2, 0, 0, 0]
        ])
        
        self.B = np.array([
            [0, 0, 0], [0, 0, 0], [0, 0, 0],
            [1, 0, 0], [0, 1, 0], [0, 0, 1]
        ])
    
    def predict(self, dt, thrust=np.zeros(3)):
        """
        Prediction step: propagate state and covariance using CW dynamics
        """
        # State transition matrix (first-order approximation)
        F = np.eye(6) + dt * self.A
        
        # State prediction
        self.state = F @ self.state + dt * self.B @ thrust
        
        # Covariance prediction
        self.covariance = F @ self.covariance @ F.T + self.Q
    
    def update_range(self, range_measurement):
        """
        Update state estimate with range measurement
        """
        # Measurement model: h(x) = sqrt(x² + y² + z²)
        x, y, z = self.state[:3]
        predicted_range = np.linalg.norm(self.state[:3])
        
        # Measurement Jacobian
        H = np.zeros((1, 6))
        if predicted_range > 0:
            H[0, :3] = self.state[:3] / predicted_range
        
        # Kalman gain
        S = H @ self.covariance @ H.T + self.R_range
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Innovation
        innovation = range_measurement - predicted_range
        
        # State and covariance update
        self.state += K.flatten() * innovation
        self.covariance = (np.eye(6) - K @ H) @ self.covariance
        
        return innovation
    
    def update_angles(self, angle_measurement):
        """
        Update state estimate with angle measurements [azimuth, elevation]
        """
        x, y, z = self.state[:3]
        range_val = np.linalg.norm(self.state[:3])
        
        if range_val == 0:
            return np.zeros(2)
        
        # Predicted measurements
        pred_azimuth = np.arctan2(y, x)
        pred_elevation = np.arcsin(z / range_val)
        predicted_angles = np.array([pred_azimuth, pred_elevation])
        
        # Measurement Jacobian
        H = np.zeros((2, 6))
        r2 = x**2 + y**2
        r3 = range_val**3
        
        # Azimuth partials
        H[0, 0] = -y / r2  # ∂azimuth/∂x
        H[0, 1] = x / r2   # ∂azimuth/∂y
        H[0, 2] = 0        # ∂azimuth/∂z
        
        # Elevation partials  
        H[1, 0] = -x*z / (range_val**2 * np.sqrt(r2))
        H[1, 1] = -y*z / (range_val**2 * np.sqrt(r2))
        H[1, 2] = np.sqrt(r2) / range_val**2
        
        # Kalman gain
        S = H @ self.covariance @ H.T + self.R_angles
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # Innovation
        innovation = angle_measurement - predicted_angles
        
        # State and covariance update
        self.state += K @ innovation
        self.covariance = (np.eye(6) - K @ H) @ self.covariance
        
        return innovation
    
    def get_state(self):
        """Return current state estimate"""
        return self.state.copy()
    
    def get_covariance(self):
        """Return current covariance estimate"""
        return self.covariance.copy()


def simulate_measurement_acquisition(trajectory, times, orbital_rate=0.00113):
    """
    Simulate sensor measurements throughout docking trajectory
    """
    sensors = SensorSuite()
    filter_ekf = NavigationFilter(trajectory[0], np.diag([100, 100, 100, 1, 1, 1]), orbital_rate)
    
    # Storage for results
    measurements_history = []
    availability_history = []
    filter_states = []
    filter_covariances = []
    
    print("Simulating sensor measurements...")
    
    for i, (state, t) in enumerate(zip(trajectory, times)):
        true_position = state[:3]
        
        # Get sensor measurements
        meas = sensors.get_measurements(true_position, t)
        measurements_history.append(meas)
        
        # Check measurement availability
        avail = sensors.measurement_availability(true_position)
        availability_history.append(avail)
        
        # Filter prediction (using simple propagation)
        if i > 0:
            dt = times[i] - times[i-1]
            filter_ekf.predict(dt)
            
            # Filter update if measurements available
            if meas is not None:
                if 'range' in meas:
                    filter_ekf.update_range(meas['range'])
                if 'angles' in meas:
                    filter_ekf.update_angles(meas['angles'])
        
        filter_states.append(filter_ekf.get_state())
        filter_covariances.append(filter_ekf.get_covariance())
    
    return measurements_history, availability_history, filter_states, filter_covariances


def analyze_measurement_performance(trajectory, times, measurements_history, availability_history):
    """
    Analyze sensor performance and generate required plots
    """
    # Extract data for analysis
    true_positions = [state[:3] for state in trajectory]
    true_ranges = [np.linalg.norm(pos) for pos in true_positions]
    
    measured_ranges = []
    range_residuals = []
    camera_available = []
    
    for i, meas in enumerate(measurements_history):
        if meas and 'range' in meas:
            measured_ranges.append(meas['range'])
            range_residuals.append(meas['range'] - true_ranges[i])
        else:
            measured_ranges.append(np.nan)
            range_residuals.append(np.nan)
        
        camera_available.append(availability_history[i]['camera_available'])
    
    # Create analysis plots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Range measurements vs true range
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(times, true_ranges, 'b-', label='True Range', linewidth=2)
    ax1.plot(times, measured_ranges, 'r.', label='Measured Range', markersize=2, alpha=0.6)
    ax1.set_ylabel('Range (m)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Range Measurement Performance')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Measurement residuals
    ax2 = fig.add_subplot(2, 2, 2)
    valid_residuals = [r for r in range_residuals if not np.isnan(r)]
    valid_times = [times[i] for i, r in enumerate(range_residuals) if not np.isnan(r)]
    ax2.plot(valid_times, valid_residuals, 'g-', linewidth=1)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Range Residual (m)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Measurement Residuals')
    ax2.grid(True)
    
    # Plot 3: Measurement availability
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(times, camera_available, 'b-', linewidth=2)
    ax3.set_ylabel('Camera Available')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Sensor Availability Timeline')
    ax3.grid(True)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['No', 'Yes'])
    
    # Plot 4: Measurement noise distribution
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(valid_residuals, bins=50, alpha=0.7, edgecolor='black', density=True)
    ax4.set_xlabel('Range Residual (m)')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Measurement Noise Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Add Gaussian fit to histogram
    if len(valid_residuals) > 0:
        from scipy.stats import norm
        mu, std = norm.fit(valid_residuals)
        x = np.linspace(min(valid_residuals), max(valid_residuals), 100)
        p = norm.pdf(x, mu, std)
        ax4.plot(x, p, 'r-', linewidth=2, label=f'Gaussian fit\nμ={mu:.3f}, σ={std:.3f}')
        ax4.legend()
    
    plt.tight_layout()
    
    # Print performance statistics
    print("\n=== SENSOR PERFORMANCE ANALYSIS ===")
    print(f"Total measurement points: {len(measurements_history)}")
    print(f"Range measurements available: {len(valid_residuals)} ({len(valid_residuals)/len(measurements_history)*100:.1f}%)")
    print(f"Camera measurements available: {sum(camera_available)} ({np.mean(camera_available)*100:.1f}%)")
    
    if len(valid_residuals) > 0:
        print(f"Range measurement statistics:")
        print(f"  Mean residual: {np.mean(valid_residuals):.3f} m")
        print(f"  Std deviation: {np.std(valid_residuals):.3f} m")
        print(f"  Max residual: {max(valid_residuals):.3f} m")
        print(f"  Min residual: {min(valid_residuals):.3f} m")
    
    return fig


# Example usage and demonstration
if __name__ == "__main__":
    # Example trajectory (circular approach for demonstration)
    T_orb = 2 * np.pi / 0.00113  # Orbital period
    times = np.linspace(0, T_orb, 1000)
    
    # Simple circular approach trajectory
    trajectory = []
    for t in times:
        # Circular approach in xy-plane
        radius = 10000 * (1 - t/T_orb)  # Decreasing radius
        x = radius * np.cos(2*np.pi*t/T_orb)
        y = radius * np.sin(2*np.pi*t/T_orb)
        z = 100 * np.sin(2*np.pi*t/T_orb)  # Small out-of-plane motion
        
        vx = -10000/T_orb * np.cos(2*np.pi*t/T_orb) - radius * 2*np.pi/T_orb * np.sin(2*np.pi*t/T_orb)
        vy = -10000/T_orb * np.sin(2*np.pi*t/T_orb) + radius * 2*np.pi/T_orb * np.cos(2*np.pi*t/T_orb)
        vz = 100 * 2*np.pi/T_orb * np.cos(2*np.pi*t/T_orb)
        
        trajectory.append(np.array([x, y, z, vx, vy, vz]))
    
    # Run measurement simulation
    measurements, availability, filter_states, filter_cov = simulate_measurement_acquisition(
        trajectory, times
    )
    
    # Analyze and plot results
    fig = analyze_measurement_performance(trajectory, times, measurements, availability)
    plt.show()
    
    print("\nMilestone 3 implementation completed successfully!")