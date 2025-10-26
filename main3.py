
import numpy as np
import matplotlib.pyplot as plt

class CameraModel:
    def __init__(self):
        # Camera intrinsic parameters
        self.focal_length = 0.05      # 50mm principal distance
        self.pixel_size = 1e-5        # 10 micron pixels
        self.image_width = 1024       # pixels
        self.image_height = 1024      # pixels
        self.centre_point = (self.image_width/2, self.image_height/2)
        
        # Field of view
        self.fov_x = np.radians(60)   # 60 degree horizontal
        self.fov_y = np.radians(60)   # 60 degree vertical 
        
        # Measurement noise
        self.pixel_noise_std = 25    # add noise to pixel values
        
        # Assuming camera points along Ceres X axis (forward)
        self.pointing_direction = np.array([1, 0, 0])
    
    def is_in_field_of_view(self, relative_position, spacecraft_attitude=np.eye(3)):
        # # Transform to camera frame (camera points along body x direction)
        # camera_direction = spacecraft_attitude @ self.pointing_direction
        
        # # Normalize the relative position vector
        # range_val = np.linalg.norm(relative_position)
        # if range_val == 0:
        #     return False
            
        # iss_direction = relative_position / range_val
        
        # # Calculate angle between camera and ISS
        # cos_angle = np.dot(camera_direction, iss_direction)
        # view_angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # # Check if within field of view (half-angle)
        # max_view_angle = max(self.fov_x, self.fov_y) / 2
        # return view_angle <= max_view_angle

        # Doesn't fucking work, assume ISS is always in view
        # attitude is always assumed as identity matrix because I can't caluclate that shit
        return True

    
    def get_measurement(self, true_position, spacecraft_attitude=np.eye(3)):
        # Assume the spacecraft can always point camera at ISS
        # In reality, this would require attitude control system
        
        # Check if ISS is in field of view (simplified - always True)
        if not self.is_in_field_of_view(true_position, spacecraft_attitude):
            return None
        
        x, y, z = true_position
        
        # Pinhole projection
        range_val = np.linalg.norm(true_position)
        
        # Calculate angles to ISS
        # These will naturally put ISS near center when properly pointed
        x_angle = np.arctan2(y, x) if x != 0 else 0
        y_angle = np.arctan2(z, np.sqrt(x**2+y**2)) if (x != 0 or y != 0) else 0
        
        # Add small random offset to simulate imperfect pointing
        max_angular_offset = np.radians(2)  # Reduced to 2 degrees for more realistic results
        x_offset = np.random.uniform(-max_angular_offset, max_angular_offset)
        y_offset = np.random.uniform(-max_angular_offset, max_angular_offset)
        
        x_angle += x_offset
        y_angle += y_offset
        
        # Convert angles to image coordinates
        x_image = self.focal_length*np.tan(x_angle)
        y_image = self.focal_length*np.tan(y_angle)
        
        # Convert to pixel coordinates
        x_pixel = x_image/self.pixel_size+self.centre_point[0]
        y_pixel = y_image/self.pixel_size+self.centre_point[1]
        
        # Check if within image bounds
        if not (0 <= x_pixel < self.image_width and 0 <= y_pixel < self.image_height):
            print(f"ISS outside image bounds: ({x_pixel:.1f}, {y_pixel:.1f})")
            return None
        
        # Add Gaussian noise
        x_pixel += np.random.normal(0, self.pixel_noise_std)
        y_pixel += np.random.normal(0, self.pixel_noise_std)
        
        return np.array([x_pixel, y_pixel])
    
    def pixels_to_angles(self, pixel_coords):
        #Convert pixel coords to angles
        x_pixel, y_pixel = pixel_coords
        x_image = (x_pixel-self.centre_point[0])*self.pixel_size
        y_image = (y_pixel-self.centre_point[1])*self.pixel_size
        
        azimuth = np.arctan2(x_image, self.focal_length)
        elevation = np.arctan2(y_image, self.focal_length)
        
        return np.array([azimuth, elevation])


class RangeSensor:
    def __init__(self):
        # add range sensor noise, rnage
        self.range_noise_std = 0.1
        self.max_range = 10000
        self.min_range = 1
    
    def get_measurement(self, true_position):
        true_range = np.linalg.norm(true_position)
        
        if true_range > self.max_range or true_range < self.min_range:
            return None
            
        measured_range = true_range + np.random.normal(0, self.range_noise_std)
        return max(measured_range, 0)


class SensorSuite:
    def __init__(self):
        # initialise other sensor classes
        self.camera = CameraModel()
        self.range_sensor = RangeSensor()
        
        # Assume camera is always pointed at ISS (simplification for this project)
        self.camera_always_pointed = True
        
    def get_all_measurements(self, true_position, time):
        measurements = {
            'time': time,
            'true_position': true_position.copy(),
            'true_range': np.linalg.norm(true_position)
        }
        
        # Get range measurement
        range_meas = self.range_sensor.get_measurement(true_position)
        if range_meas is not None:
            measurements['range'] = range_meas
            measurements['range_residual'] = range_meas - measurements['true_range']
        
        # Get camera measurement - ASSUME camera can always point at ISS
        if self.camera_always_pointed:
            pixel_meas = self.camera.get_measurement(true_position)
            if pixel_meas is not None:
                measurements['pixels'] = pixel_meas
                measurements['angles'] = self.camera.pixels_to_angles(pixel_meas)
                print(f"Camera measurement: ISS at pixels ({pixel_meas[0]:.1f}, {pixel_meas[1]:.1f})")
        
        if 'range' not in measurements and 'pixels' not in measurements:
            return None
            
        return measurements
    
    def check_availability(self, true_position):
        range_avail = self.range_sensor.get_measurement(true_position) is not None
        
        # Camera availability: assume always available if ISS in range
        camera_avail = (range_avail and self.camera_always_pointed)
        
        return {
            'range_available': range_avail,
            'camera_available': camera_avail
        }


def simulate_sensor_measurements(trajectory, times):
    sensors = SensorSuite()
    
    measurements_history = []
    availability_history = []
    
    print("Simulating sensor measurements with corrected camera model...")
    
    for i, (state, t) in enumerate(zip(trajectory, times)):
        true_position = state[:3]
        
        meas = sensors.get_all_measurements(true_position, t)
        measurements_history.append(meas)
        
        avail = sensors.check_availability(true_position)
        availability_history.append(avail)
        
        if i % 200 == 0:
            range_avail = avail['range_available']
            camera_avail = avail['camera_available']
            range_val = np.linalg.norm(true_position)
            has_pixels = meas is not None and 'pixels' in meas
            print(f"Time {t:.1f}s: Range={range_val:.0f}m, RangeAvail={range_avail}, CameraAvail={camera_avail}, HasPixels={has_pixels}")
    
    return measurements_history, availability_history


def analyse_measurement_performance(trajectory, times, measurements_history, availability_history):
    true_positions = [state[:3] for state in trajectory]
    true_ranges = [np.linalg.norm(pos) for pos in true_positions]
    
    measured_ranges = []
    range_residuals = []
    camera_available = []
    range_available = []
    pixel_data = []
    
    for i, meas in enumerate(measurements_history):
        if meas and 'range' in meas:
            measured_ranges.append(meas['range'])
            range_residuals.append(meas['range_residual'])
        else:
            measured_ranges.append(np.nan)
            range_residuals.append(np.nan)
        
        camera_available.append(availability_history[i]['camera_available'])
        range_available.append(availability_history[i]['range_available'])
        
        # Track pixel data
        if meas and 'pixels' in meas:
            pixel_data.append(meas['pixels'])
        else:
            pixel_data.append(None)
    
    # Print pixel statistics
    valid_pixels = [p for p in pixel_data if p is not None]
    print(f"\nPixel data: {len(valid_pixels)} valid measurements out of {len(pixel_data)}")
    if valid_pixels:
        avg_x = np.mean([p[0] for p in valid_pixels])
        avg_y = np.mean([p[1] for p in valid_pixels])
        print(f"Average pixel position: ({avg_x:.1f}, {avg_y:.1f})")
    
    # Create plots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Range measurements
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
    ax2.plot(valid_times, valid_residuals, 'g-', linewidth=1, alpha=0.7)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Range Residual (m)')
    ax2.set_xlabel('Time (s)')
    ax2.set_title('Measurement Residuals')
    ax2.grid(True)
    
    # Plot 3: Sensor availability
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(times, range_available, 'b-', label='Range Sensor', linewidth=2)
    ax3.plot(times, camera_available, 'r-', label='Camera', linewidth=2)
    ax3.set_ylabel('Sensor Available')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Sensor Availability Timeline')
    ax3.legend()
    ax3.grid(True)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['No', 'Yes'])
    
    # Plot 4: Noise distribution
    ax4 = fig.add_subplot(2, 2, 4)
    if valid_residuals:
        ax4.hist(valid_residuals, bins=30, alpha=0.7, edgecolor='black', density=True)
        from scipy.stats import norm
        mu, std = norm.fit(valid_residuals)
        x = np.linspace(min(valid_residuals), max(valid_residuals), 100)
        p = norm.pdf(x, mu, std)
        ax4.plot(x, p, 'r-', linewidth=2, label=f'Gaussian fit\nμ={mu:.3f}, σ={std:.3f}')
        ax4.set_xlabel('Range Residual (m)')
        ax4.set_ylabel('Probability Density')
        ax4.set_title('Measurement Noise Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print statistics
    print("\n" + "="*50)
    print("CORRECTED SENSOR PERFORMANCE ANALYSIS")
    print("="*50)
    print(f"Total points: {len(trajectory)}")
    print(f"Range measurements: {len(valid_residuals)} ({len(valid_residuals)/len(trajectory)*100:.1f}%)")
    print(f"Camera measurements: {sum(camera_available)} ({np.mean(camera_available)*100:.1f}%)")
    
    if valid_residuals:
        print(f"\nRange Sensor Performance:")
        print(f"  Mean residual: {np.mean(valid_residuals):.4f} m")
        print(f"  Standard deviation: {np.std(valid_residuals):.4f} m")
    
    return fig

def generate_image(pixel_coords, image_width=1024, image_height=1024, pixel_noise_std=25):
    # Create grid with camera measurement noise for each pixel
    # Start with base value of 0.5 (medium gray)
    base_value = 0.5
    grid = np.ones((image_height, image_width, 3)) * base_value
    
    # Add camera measurement noise to each pixel
    noise = np.random.normal(0, pixel_noise_std/255.0, (image_height, image_width, 3))
    grid += noise
    
    # make sure the noise is between 0 and 1
    grid = np.clip(grid, 0, 1)
    
    if pixel_coords is not None:
        x, y = pixel_coords.astype(int)
        if 0 <= x < image_width and 0 <= y < image_height:
            # Create a 5x5 red square for ISS
            square_size = 5
            for dx in range(-square_size//2, square_size//2 + 1):
                for dy in range(-square_size//2, square_size//2 + 1):
                    px = x + dx
                    py = y + dy
                    if 0 <= px < image_width and 0 <= py < image_height:
                        grid[py, px] = [1, 0, 0]  # Red pixel
    
    return grid

def display_image(measurements_history, trajectory, times, target_time=0):
    # Find the index closest to target_time
    time = np.argmin(np.abs(times - target_time))
    
    meas = measurements_history[time]
    true_position = trajectory[time][:3]
    true_range = np.linalg.norm(true_position)
    time_val = times[time]
    
    # Get pixel coordinates
    pixel_coords = meas['pixels'] if meas and 'pixels' in meas else None
    
    print(f"Time: {time_val:.1f}s, Range: {true_range:.0f}m")
    print(f"ISS pixel coordinates: {pixel_coords}")
    print(f"Using camera measurement noise: std={0.5} pixels")
    
    # Generate an image
    grid = generate_image(pixel_coords)
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Display the grid
    ax.imshow(grid)
    ax.set_title(f'Camera View at t=0\nRange: {true_range:.0f}m | Pixel noise: σ=25', fontsize=14)
    ax.set_xlabel('X pixels')
    ax.set_ylabel('Y pixels')
    
    # Add grid lines every 100 pixels for reference
    ax.set_xticks(np.arange(0, 1024, 100))
    ax.set_yticks(np.arange(0, 1024, 100))
    ax.grid(True, alpha=0.3)
    
    # Mark center with crosshairs
    center_x, center_y = 512, 512
    ax.axhline(y=center_y, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(x=center_x, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    ax.plot(center_x, center_y, 'b+', markersize=15, markeredgewidth=3)
    ax.text(center_x + 10, center_y + 10, 'Center', color='blue', fontweight='bold')
    
    # Mark ISS position if visible
    if pixel_coords is not None:
        iss_x, iss_y = pixel_coords.astype(int)
        
        # Calculate offset from center
        dx = iss_x - center_x
        dy = iss_y - center_y
        ax.text(10, 30, f'ISS Offset: ({dx}, {dy}) pixels', color='red', 
               bbox=dict(facecolor='white', alpha=0.8), fontweight='bold', fontsize=11)
        
        # Show actual pixel values
        ax.text(10, 60, f'ISS Pixel: ({iss_x}, {iss_y})', color='red',
               bbox=dict(facecolor='white', alpha=0.8), fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    return fig

# Test with realistic trajectory
if __name__ == "__main__":
    # Create a more realistic approach trajectory
    T_orb = 2 * np.pi / 0.00113
    times = np.linspace(0, T_orb, 1000)
    trajectory = []
    
    for t in times:
        # Spiral in toward ISS
        progress = t / T_orb
        radius = 10000 * (1 - progress)**2  
        angle = 4 * np.pi * progress
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 100 * np.sin(2 * angle)
        
        # Velocities for finding state vector
        vx = -radius/T_orb*2*(1-progress)*np.cos(angle)-radius*4*np.pi/T_orb*np.sin(angle)
        vy = -radius/T_orb*2*(1-progress)*np.sin(angle)+radius*4*np.pi/T_orb*np.cos(angle)
        vz = 100*4*np.pi/T_orb*np.cos(2*angle)
        
        trajectory.append(np.array([x, y, z, vx, vy, vz]))
    
    measurements, availability = simulate_sensor_measurements(trajectory, times)
    fig = analyse_measurement_performance(trajectory, times, measurements, availability)
    plt.show()
    
    # Display SINGLE pixel grid at t=0 using camera measurement noise
    single_grid_fig = display_image(measurements, trajectory, times, target_time=0)
    plt.show()