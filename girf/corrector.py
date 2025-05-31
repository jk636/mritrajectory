import numpy as np
# Attempt to import nibabel for NIFTI, but make it optional
try:
    import nibabel as nib
except ImportError:
    nib = None
# Attempt to import matplotlib for simple image saving, also optional
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class TrajectoryCorrector:
    def __init__(self, reconstruction_config=None):
        """
        Initializes the TrajectoryCorrector.

        Args:
            reconstruction_config (dict, optional): Configuration for image reconstruction.
                Example: {'matrix_size': (256, 256), 'fov': (0.256, 0.256)} # FOV in meters
        """
        default_config = {
            'matrix_size': (128, 128), # Default image matrix size
            'fov': (0.2, 0.2),         # Default Field of View in meters (e.g., 20cm x 20cm)
            'gridding_oversampling_factor': 1.0, # No oversampling by default for simple gridding
            'kernel_name': 'nearest' # Default simple gridding
        }
        if reconstruction_config is None:
            self.reconstruction_config = default_config
        else:
            self.reconstruction_config = {**default_config, **reconstruction_config}

        self.predicted_trajectory_kspace = None # Corrected k-space coordinates
        self.raw_kspace_samples = None      # Acquired k-space data
        self.gridded_kspace = None          # K-space data on Cartesian grid
        self.image_data = None              # Reconstructed image

        print(f"TrajectoryCorrector initialized with config: {self.reconstruction_config}")

    def _prepare_trajectory_for_gridding(self, trajectory_data):
        """Converts various trajectory formats to a standard (N_points, N_dims) NumPy array."""
        if isinstance(trajectory_data, dict):
            # Assuming keys like 'x', 'y', 'z' or 'kx', 'ky', 'kz'
            # Try to determine order, simple sort for now
            axes_keys = sorted(trajectory_data.keys())
            if not axes_keys:
                raise ValueError("Trajectory dictionary is empty.")

            # Check lengths are consistent
            data_len = -1
            for key in axes_keys:
                current_len = len(trajectory_data[key])
                if data_len == -1: data_len = current_len
                elif data_len != current_len:
                    raise ValueError("Inconsistent lengths in trajectory dictionary.")

            dims = len(axes_keys)
            np_trajectory = np.zeros((data_len, dims))
            for i, key in enumerate(axes_keys):
                np_trajectory[:, i] = trajectory_data[key]
            return np_trajectory
        elif isinstance(trajectory_data, (list, np.ndarray)):
            np_trajectory = np.asarray(trajectory_data)
            if np_trajectory.ndim == 1: # Single axis trajectory
                np_trajectory = np_trajectory[:, np.newaxis]
            if np_trajectory.ndim != 2:
                raise ValueError("Trajectory data must be convertible to 2D (num_points, num_axes).")
            return np_trajectory
        else:
            raise TypeError("Unsupported trajectory data type. Must be dict, list, or NumPy array.")


    def reconstruct_image(self, predicted_kspace_trajectory, raw_kspace_samples):
        """
        Reconstructs an image from k-space data using the predicted trajectory.
        Uses simplified nearest-neighbor gridding.

        Args:
            predicted_kspace_trajectory (np.array or dict): The GIRF-corrected k-space trajectory coordinates.
                                               Expected shape (N_points, N_dims) or dict {'x':[], 'y':[]}.
                                               Coordinates are in m^-1.
            raw_kspace_samples (np.array or list): The acquired k-space data samples (complex-valued).
                                              Expected shape (N_points,).

        Returns:
            np.array: The reconstructed image data (complex-valued before abs, real-valued after).
        """
        self.predicted_trajectory_kspace = self._prepare_trajectory_for_gridding(predicted_kspace_trajectory)
        self.raw_kspace_samples = np.asarray(raw_kspace_samples)

        if self.predicted_trajectory_kspace.shape[0] != len(self.raw_kspace_samples):
            raise ValueError("Mismatch between number of trajectory points and k-space samples. "
                             f"Got {self.predicted_trajectory_kspace.shape[0]} points and {len(self.raw_kspace_samples)} samples.")

        num_dims = self.predicted_trajectory_kspace.shape[1]
        if num_dims not in [2, 3]:
            raise ValueError(f"Trajectory must be 2D or 3D. Got {num_dims} dimensions.")

        matrix_size = self.reconstruction_config.get('matrix_size', (128, 128) if num_dims == 2 else (64, 64, 64))
        fov = self.reconstruction_config.get('fov', (0.2, 0.2) if num_dims == 2 else (0.2, 0.2, 0.2)) # meters

        if len(matrix_size) != num_dims or len(fov) != num_dims:
            raise ValueError(f"matrix_size ({matrix_size}) and fov ({fov}) must match trajectory dimensions ({num_dims}).")

        print(f"Starting image reconstruction. Matrix size: {matrix_size}, FOV: {fov} m, Num K-samples: {len(self.raw_kspace_samples)}")

        # Initialize k-space grid
        self.gridded_kspace = np.zeros(matrix_size, dtype=np.complex128)
        density_compensation = np.zeros(matrix_size, dtype=np.float64) # For simple density counting

        # Scale k-space coordinates to grid indices
        # k_max for each dimension: 1 / (2 * pixel_size) = matrix_size / (2 * FOV)
        # Grid index = k_coord * (FOV / matrix_size_coord) * matrix_size_coord + matrix_size_coord / 2
        # Simplified: index = k_coord * FOV_dim + matrix_size_dim / 2 (if k is normalized to [-0.5, 0.5]*k_max_ fysische eenheden)
        # k_abs_max = [ms / (2 * f) for ms, f in zip(matrix_size, fov)] # Max k value that fits in grid (Nyquist)

        # Scaling factor to map k-space coordinates (in m^-1) to grid indices
        # k_idx = k_val * fov_val + matrix_size_val / 2 (Incorrect, this is for k in cycles/FOV)
        # Correct scaling: k_idx = k_val * (grid_size_val * pixel_size_val) + grid_center_idx
        # k_idx = k_val * fov_val + matrix_size_val / 2 (if k normalized from -kmax/2 to kmax/2)
        # k_idx = k_val * (matrix_size_val / (k_nyquist_val * 2)) + matrix_size_val / 2
        # k_nyquist = 1 / (2 * pixel_size) = matrix_size / (2 * fov)
        # k_idx = k_val / (2 * k_nyquist_val) * matrix_size_val + matrix_size_val / 2
        # k_idx = k_val * fov_val / matrix_size_val * matrix_size_val + matrix_size_val / 2 (still not right)
        # k_idx = (k_val * fov_dim) + matrix_size_dim / 2 (if k is normalized to [-0.5, 0.5] cycles/FOV)

        # Let k_norm = k_val / k_max_physical, where k_max_physical maps to edge of grid.
        # k_max_physical for dimension i is matrix_size[i] / (2 * fov[i]) (Nyquist frequency)
        # So, k_scaled_to_grid_units = k_val / (1/fov[i]) = k_val * fov[i]
        # Then map this to [-matrix_size/2, matrix_size/2]
        # grid_indices = np.round(k_scaled_to_grid_units + np.array(matrix_size)/2).astype(int)

        # Simplified scaling based on typical MRI conventions where k-space extent is 1/pixel_size
        # k_max_grid = matrix_size / 2
        # k_coords_scaled_to_grid = (predicted_kspace_trajectory / k_nyquist_for_each_axis) * k_max_grid_for_each_axis
        # k_coords_scaled_to_grid = predicted_kspace_trajectory * fov (if k was normalized to cycles/grid unit)

        # Correct scaling:
        # k_grid_indices = k_physical * (grid_dimension_size / k_max_physical_range)
        # k_max_physical_range for dimension 'd' is matrix_size[d] / fov[d] (representing total k-space width in m^-1)
        # So, k_idx = k_physical_val * (matrix_size[d] / (matrix_size[d]/fov[d])) = k_physical_val * fov[d]
        # This maps k_physical to units of "grid cells if FOV was 1m".
        # Then shift origin: k_idx_shifted = k_idx + matrix_size[d]/2

        grid_indices = np.zeros_like(self.predicted_trajectory_kspace, dtype=int)
        for dim_idx in range(num_dims):
            # Scale factor maps physical k-space units (m^-1) to grid index units.
            # Full k-space width covered by grid: matrix_size[dim_idx] / fov[dim_idx] (in m^-1)
            # Map k_coord from [-k_max_phys/2, k_max_phys/2] to [0, matrix_size-1]
            # k_norm = (k_coord / (matrix_size[dim_idx]/fov[dim_idx])) + 0.5
            # index = k_norm * matrix_size[dim_idx]
            scale = matrix_size[dim_idx] / (matrix_size[dim_idx] / fov[dim_idx]) # This is just fov[dim_idx]
            grid_indices[:, dim_idx] = np.round(
                self.predicted_trajectory_kspace[:, dim_idx] * scale + matrix_size[dim_idx] / 2.0
            ).astype(int)


        # Perform nearest-neighbor gridding
        for i in range(len(self.raw_kspace_samples)):
            idx_tuple = tuple(grid_indices[i, dim] for dim in range(num_dims))

            # Check if indices are within grid boundaries
            valid_indices = True
            for dim_idx in range(num_dims):
                if not (0 <= idx_tuple[dim_idx] < matrix_size[dim_idx]):
                    valid_indices = False
                    break

            if valid_indices:
                self.gridded_kspace[idx_tuple] += self.raw_kspace_samples[i]
                density_compensation[idx_tuple] += 1 # Simple density: count hits per cell

        # Density compensation (very basic: divide by number of hits)
        # Avoid division by zero for cells with no hits.
        density_compensation[density_compensation == 0] = 1 # Avoid div by zero, effectively no change for empty cells
        self.gridded_kspace /= density_compensation
        print("Gridding complete (Nearest Neighbor).")

        # Perform Inverse FFT
        # Shift k-space center before IFFT, then shift back for image
        if num_dims == 2:
            shifted_kspace = np.fft.ifftshift(self.gridded_kspace)
            image_shifted_back = np.fft.fftshift(np.fft.ifft2(shifted_kspace))
        elif num_dims == 3:
            shifted_kspace = np.fft.ifftshift(self.gridded_kspace)
            image_shifted_back = np.fft.fftshift(np.fft.ifftn(shifted_kspace))
        else: # Should not happen due to earlier check
            raise NotImplementedError("Only 2D and 3D reconstruction supported.")

        self.image_data = image_shifted_back
        print("Image reconstruction complete (IFFT).")
        return self.image_data


    def evaluate_image_quality(self, metrics_config=None):
        """
        Evaluates the quality of the reconstructed image. (Placeholder)

        Args:
            metrics_config (dict, optional): Configuration for metrics calculation.
                                            (e.g., {'roi_signal': [slice,row_start,row_end,col_start,col_end],
                                                    'roi_noise': [...]})
        Returns:
            dict: Dictionary of computed quality metrics.
        """
        if self.image_data is None:
            print("Warning: Image data not available. Cannot evaluate quality.")
            return {"error": "Image not reconstructed"}

        # Placeholder for actual image quality metrics
        # SNR: requires defining signal and noise regions.
        # Sharpness: e.g., gradient magnitude, edge analysis.
        # For now, return some basic stats.
        img_abs = np.abs(self.image_data)
        computed_metrics = {
            'mean_intensity': np.mean(img_abs),
            'max_intensity': np.max(img_abs),
            'std_intensity': np.std(img_abs),
            'snr_placeholder': None, # Requires noise estimation
            'sharpness_placeholder': None
        }
        print(f"Image quality metrics (simulated): {computed_metrics}")
        return computed_metrics

    def save_image(self, file_path, file_format='NPY'):
        """
        Saves the reconstructed image to a file.

        Args:
            file_path (str): The path to save the image file.
            file_format (str, optional): Format to save in ('NPY', 'NIFTI', 'PNG'). Defaults to 'NPY'.
        """
        if self.image_data is None:
            print("Error: No image data to save.")
            return

        print(f"Saving image to {file_path} in {file_format} format...")
        try:
            if file_format.upper() == 'NPY':
                np.save(file_path, np.abs(self.image_data)) # Save magnitude for NPY
                print(f"Image saved successfully as NumPy array: {file_path}.npy")
            elif file_format.upper() == 'NIFTI':
                if nib is None:
                    print("Error: Nibabel library not installed. Cannot save as NIFTI.")
                    return
                # Create a Nifti1Image object. Requires an affine matrix.
                # For simplicity, use identity affine. A proper affine would come from scanner geometry.
                # Data should be suitably oriented (e.g. RAS+) and typically real-valued (magnitude or phase).
                img_to_save = np.abs(self.image_data).astype(np.float32) # Magnitude image
                # Ensure it's at least 3D for NIFTI
                if img_to_save.ndim == 2:
                    img_to_save = img_to_save[:, :, np.newaxis]

                affine = np.eye(4) # Placeholder affine
                # Set pixel dimensions from FOV and matrix size if possible
                # Assuming 2D for simplicity here if FOV and matrix_size are 2D
                if len(self.reconstruction_config['fov']) >=2 and len(self.reconstruction_config['matrix_size']) >=2:
                    pixdim_x = self.reconstruction_config['fov'][0] / self.reconstruction_config['matrix_size'][0]
                    pixdim_y = self.reconstruction_config['fov'][1] / self.reconstruction_config['matrix_size'][1]
                    affine[0,0] = pixdim_x * 1000 # Convert m to mm for typical NIFTI pixdim
                    affine[1,1] = pixdim_y * 1000
                if img_to_save.ndim == 3 and len(self.reconstruction_config['fov']) ==3 and len(self.reconstruction_config['matrix_size']) ==3 :
                     pixdim_z = self.reconstruction_config['fov'][2] / self.reconstruction_config['matrix_size'][2]
                     affine[2,2] = pixdim_z * 1000

                nifti_img = nib.Nifti1Image(img_to_save, affine)
                nib.save(nifti_img, file_path)
                print(f"Image saved successfully as NIFTI: {file_path}")
            elif file_format.upper() == 'PNG':
                if plt is None:
                    print("Error: Matplotlib library not installed. Cannot save as PNG.")
                    return
                # Save magnitude, normalized, as grayscale PNG
                img_abs = np.abs(self.image_data)
                norm_img = (img_abs - np.min(img_abs)) / (np.max(img_abs) - np.min(img_abs) + 1e-9) # Normalize to [0,1]
                plt.imsave(file_path, norm_img, cmap='gray')
                print(f"Image saved successfully as PNG: {file_path}")
            else:
                print(f"Error: Unsupported file format '{file_format}'. Supported: NPY, NIFTI, PNG.")
        except Exception as e:
            print(f"Error saving image to {file_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    print("--- Running TrajectoryCorrector Example with Actual Logic ---")

    # 1. Reconstruction Configuration
    matrix_s = 64
    fov_m = 0.2 # 200 mm
    config = {
        'matrix_size': (matrix_s, matrix_s),
        'fov': (fov_m, fov_m) # meters
    }
    corrector = TrajectoryCorrector(reconstruction_config=config)

    # 2. Dummy Data Generation
    num_samples = 1000 # Number of k-space samples

    # Dummy Predicted K-space Trajectory (k-space coordinates in m^-1)
    # Simulate a spiral-like trajectory for 2D
    k_max_phys = matrix_s / (2 * fov_m) # Physical k-space extent covered by grid (Nyquist)

    theta = np.linspace(0, 10 * np.pi, num_samples)
    radius = np.linspace(0, 0.5 * k_max_phys, num_samples) # Go up to half Nyquist for this example

    dummy_predicted_kx = radius * np.cos(theta)
    dummy_predicted_ky = radius * np.sin(theta)
    dummy_trajectory = np.stack([dummy_predicted_kx, dummy_predicted_ky], axis=-1)
    # Or as dict: dummy_trajectory_dict = {'kx': dummy_predicted_kx, 'ky': dummy_predicted_ky}

    # Dummy Raw K-space Samples (complex values)
    # Simulate signal from a point object at the center (whose FT is constant across k-space)
    # Add some noise.
    dummy_raw_data = np.ones(num_samples, dtype=np.complex128) * 100
    dummy_raw_data += np.random.normal(0, 10, num_samples) + 1j * np.random.normal(0, 10, num_samples)


    # 3. Reconstruct Image
    try:
        reconstructed_img = corrector.reconstruct_image(dummy_trajectory, dummy_raw_data)
        # reconstructed_img_from_dict = corrector.reconstruct_image(dummy_trajectory_dict, dummy_raw_data)
        print(f"Reconstructed image shape: {reconstructed_img.shape}")
        # print(f"Gridded k-space center value: {corrector.gridded_kspace[matrix_s//2, matrix_s//2]}")

        # 4. Evaluate Image Quality (placeholder)
        quality_metrics = corrector.evaluate_image_quality()
        # print(f"Image quality: {quality_metrics}")

        # 5. Save Image
        corrector.save_image("dummy_reconstructed_image.npy", format='NPY')
        # corrector.save_image("dummy_reconstructed_image.nii.gz", format='NIFTI') # Needs nibabel
        # corrector.save_image("dummy_reconstructed_image.png", format='PNG') # Needs matplotlib

        # Optional: Display with matplotlib if available
        if plt:
            plt.figure(figsize=(6,6))
            plt.imshow(np.abs(reconstructed_img), cmap='gray')
            plt.title("Reconstructed Image (Magnitude)")
            plt.colorbar()
            # plt.savefig("dummy_reconstructed_image_display.png")
            print("Displayed image using matplotlib (if plt.show() is called or in interactive mode).")


    except ValueError as e:
        print(f"Error in reconstruction pipeline: {e}")
    except Exception as e_gen:
        print(f"An unexpected error occurred: {e_gen}")
        import traceback
        traceback.print_exc()

    print("\n--- TrajectoryCorrector Example Finished ---")
