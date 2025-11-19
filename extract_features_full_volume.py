"""
Advanced feature extraction that captures and combines features from ALL sliding window patches.
This gives you features corresponding to the full input volume (all slices).

Extracts THREE types of features:
1. feature_neg2: decoder.stages.5.convs.0.conv (second-to-last conv, before batch norm/activation)
2. feature_neg1: decoder.stages.5.convs.1.conv (last conv before logits, before batch norm/activation)
3. logits: decoder.seg_layers.5 (segmentation head output)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple
import os

# Patch torch.load to handle PyTorch 2.6+ weights_only default
original_torch_load = torch.load
def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)

torch.load = patched_torch_load


class SlidingWindowFeatureExtractor:
    """
    Advanced feature extractor that captures features from all sliding window patches
    and reconstructs the full-volume features.
    """
    
    def __init__(self, model: nn.Module, target_layers: List[str] = None):
        self.model = model
        self.target_layers = target_layers or self._find_target_layers()
        self.patch_features = []  # Store features from all patches
        self.patch_coordinates = []  # Store patch positions
        self.hooks = []
        self.current_patch_idx = 0
        self._register_hooks()
    
    def _find_target_layers(self) -> List[str]:
        """Find the target layers for feature extraction."""
        target_layers = []
        named_modules = dict(self.model.named_modules())
        
        # Find decoder.stages.5 (final processed features)
        decoder_stages = []
        for name in named_modules.keys():
            if 'decoder' in name and 'stages' in name and len(name.split('.')) == 3:
                try:
                    stage_num = int(name.split('.')[-1])
                    decoder_stages.append((stage_num, name))
                except ValueError:
                    continue
        
        if decoder_stages:
            last_stage = max(decoder_stages, key=lambda x: x[0])
            
            # Extract features from conv layers before batch norm and activation
            # decoder.stages.5.convs.0.conv - Second-to-last convolution
            conv0_layer = f"{last_stage[1]}.convs.0.conv"
            if conv0_layer in named_modules:
                target_layers.append(conv0_layer)
            
            # decoder.stages.5.convs.1.conv - Last convolution before logits
            conv1_layer = f"{last_stage[1]}.convs.1.conv"
            if conv1_layer in named_modules:
                target_layers.append(conv1_layer)
        
        # Also extract logits from segmentation head
        # Find decoder.seg_layers.5 (or the last seg_layer)
        seg_layers = []
        for name in named_modules.keys():
            if 'decoder.seg_layers' in name and len(name.split('.')) == 3:
                try:
                    seg_num = int(name.split('.')[-1])
                    seg_layers.append((seg_num, name))
                except ValueError:
                    continue
        
        if seg_layers:
            last_seg_layer = max(seg_layers, key=lambda x: x[0])
            target_layers.append(last_seg_layer[1])  # decoder.seg_layers.5
        
        print(f"Target layers for full-volume feature extraction: {target_layers}")
        return target_layers
    
    def _register_hooks(self):
        """Register hooks that capture features from each patch."""
        def hook_fn(name):
            def hook(module, input, output):
                # Store features from this patch with metadata
                # Create a more descriptive layer name for clarity
                if 'convs.0.conv' in name:
                    layer_desc = 'feature_neg2'
                elif 'convs.1.conv' in name:
                    layer_desc = 'feature_neg1'
                elif 'seg_layers' in name:
                    layer_desc = 'logits'
                else:
                    layer_desc = name.replace('.', '_')
                
                patch_info = {
                    'layer_name': layer_desc,
                    'original_layer_name': name,
                    'features': output.detach().clone().cpu(),  # Move to CPU immediately
                    'patch_idx': self.current_patch_idx,
                    'shape': output.shape
                }
                self.patch_features.append(patch_info)
                print(f"Captured patch {self.current_patch_idx} features from {layer_desc}: {output.shape}")
            return hook
        
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
                print(f"Registered sliding window hook on: {name}")
                if 'convs.0.conv' in name:
                    print(f"  → This extracts feature_neg2 (second-to-last conv before batch norm/activation)")
                elif 'convs.1.conv' in name:
                    print(f"  → This extracts feature_neg1 (last conv before logits, before batch norm/activation)")
                elif 'seg_layers' in name:
                    print(f"  → This extracts logits (segmentation head output)")
    
    def reset_for_new_case(self):
        """Reset state for processing a new case."""
        self.patch_features = []
        self.patch_coordinates = []
        self.current_patch_idx = 0
    
    def increment_patch(self):
        """Call this when moving to next patch."""
        self.current_patch_idx += 1
    
    def combine_patch_features(self, original_shape: Tuple[int, ...], patch_size: Tuple[int, ...], 
                             step_size: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        Combine features from all patches back into full-volume features.
        
        Args:
            original_shape: Shape of the original input (e.g., [39, 256, 256])
            patch_size: Size of patches used (e.g., [28, 256, 256])
            step_size: Step size used in sliding window
            
        Returns:
            Combined features for the full volume
        """
        if not self.patch_features:
            return {}
        
        # Group features by layer
        features_by_layer = {}
        for patch_info in self.patch_features:
            layer_name = patch_info['layer_name']
            if layer_name not in features_by_layer:
                features_by_layer[layer_name] = []
            features_by_layer[layer_name].append(patch_info)
        
        combined_features = {}
        
        for layer_name, patch_list in features_by_layer.items():
            print(f"\nCombining {len(patch_list)} patches for layer: {layer_name}")
            
            # Get feature dimensions (features are typically downsampled from input)
            sample_features = patch_list[0]['features']
            batch_size, channels = sample_features.shape[0], sample_features.shape[1]
            feature_spatial_shape = sample_features.shape[2:]  # Spatial dimensions of features
            
            # Calculate downsampling factor between input and features
            downsample_factors = [orig // feat for orig, feat in zip(patch_size, feature_spatial_shape)]
            
            # Calculate output feature volume size
            output_feature_shape = [orig // ds for orig, ds in zip(original_shape, downsample_factors)]
            
            print(f"  Layer: {layer_name}")
            print(f"  Input shape: {original_shape}")
            print(f"  Patch size: {patch_size}")
            print(f"  Feature channels: {channels}")
            print(f"  Feature patch shape: {feature_spatial_shape}")
            print(f"  Downsample factors: {downsample_factors}")
            print(f"  Output feature shape: {output_feature_shape}")
            
            # Initialize output tensor and weight tensor for averaging
            full_features = torch.zeros([batch_size, channels] + output_feature_shape, 
                                       dtype=sample_features.dtype, device='cpu')
            weight_map = torch.zeros(output_feature_shape, dtype=torch.float32, device='cpu')
            
            # Calculate patch positions in the original volume
            step_in_voxels = int(patch_size[0] * step_size)  # Step size in voxels
            patch_positions = []
            
            start_pos = 0
            while start_pos < original_shape[0]:
                end_pos = min(start_pos + patch_size[0], original_shape[0])
                if end_pos - start_pos < patch_size[0]:
                    # Last patch: adjust start position
                    start_pos = max(0, original_shape[0] - patch_size[0])
                    end_pos = original_shape[0]
                
                patch_positions.append((start_pos, end_pos))
                
                if end_pos >= original_shape[0]:
                    break
                start_pos += step_in_voxels
            
            print(f"  Calculated patch positions: {patch_positions}")
            print(f"  Number of patches: {len(patch_positions)} (captured: {len(patch_list)})")
            
            # Place each patch's features in the correct position
            for i, (patch_info, (start_pos, end_pos)) in enumerate(zip(patch_list, patch_positions)):
                features = patch_info['features']
                
                # Calculate position in feature space
                feat_start = start_pos // downsample_factors[0]
                feat_end = end_pos // downsample_factors[0]
                
                # Handle case where feature patch might be larger than calculated space
                feat_patch_size = features.shape[2]  # First spatial dimension of features
                feat_end = min(feat_start + feat_patch_size, output_feature_shape[0])
                
                print(f"    Patch {i}: input[{start_pos}:{end_pos}] -> features[{feat_start}:{feat_end}]")
                
                # Add features to output (weighted for overlapping regions)
                if feat_end > feat_start:
                    actual_feat_size = feat_end - feat_start
                    full_features[:, :, feat_start:feat_end, :, :] += features[:, :, :actual_feat_size, :, :]
                    weight_map[feat_start:feat_end, :, :] += 1.0
            
            # Average overlapping regions
            weight_map[weight_map == 0] = 1.0  # Avoid division by zero
            full_features = full_features / weight_map.unsqueeze(0).unsqueeze(0)
            
            combined_features[layer_name] = full_features
            
            print(f"  Final combined features shape: {full_features.shape}")
        
        return combined_features
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def extract_full_volume_features(model_folder: str,
                                input_files: List[str],
                                device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    Extract features for the FULL input volume by capturing and combining
    features from all sliding window patches.
    
    Returns features that correspond to the complete input (e.g., all 39 slices).
    """
    
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    
    print("="*60)
    print("FULL VOLUME CONVOLUTION FEATURE EXTRACTION")
    print("="*60)
    print("Extracting features from:")
    print("  1. decoder.stages.5.convs.0.conv (feature_neg2: second-to-last convolution)")
    print("  2. decoder.stages.5.convs.1.conv (feature_neg1: last convolution before logits)")
    print("  3. decoder.seg_layers.5 (logits: segmentation head output)")
    print("  Features 1&2 extracted BEFORE batch normalization and activation")
    print("="*60)
    
    predictor = nnUNetPredictor(
        device=torch.device(device),
        perform_everything_on_device=True,
        verbose=False,
        use_gaussian=True,
        use_mirroring=False,
        tile_step_size=0.5  # This determines patch overlap
    )
    
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=None)
    
    # Get model configuration
    patch_size = predictor.configuration_manager.patch_size
    print(f"Model patch size: {patch_size}")
    
    # Load input data to get original dimensions
    rw = SimpleITKIO()
    images = []
    for file_path in input_files:
        img, props = rw.read_images([file_path])
        images.append(img[0])
    
    image_data = np.stack(images, axis=0)
    original_shape = image_data.shape[1:]  # Skip channel dimension
    print(f"Original input shape: {original_shape}")
    
    # Create sliding window feature extractor
    extractor = SlidingWindowFeatureExtractor(predictor.network)
    extractor.reset_for_new_case()
    
    # Monkey patch the sliding window prediction to track patches
    original_predict_method = predictor._internal_maybe_mirror_and_predict
    
    def patched_predict_method(x):
        """Modified prediction that tracks patch processing."""
        result = original_predict_method(x)
        extractor.increment_patch()
        return result
    
    predictor._internal_maybe_mirror_and_predict = patched_predict_method
    
    try:
        print("\nRunning sliding window prediction to capture all patch features...")
        
        # Run prediction - this will capture features from ALL patches
        result = predictor.predict_single_npy_array(
            image_data,
            image_properties={'spacing': props['spacing']},
            segmentation_previous_stage=None,
            output_file_truncated=None,
            save_or_return_probabilities=False
        )
        
        print(f"\nPrediction completed. Captured features from {len(extractor.patch_features)} patches.")
        
        # Combine features from all patches
        print("\nCombining patch features into full-volume features...")
        full_volume_features = extractor.combine_patch_features(
            original_shape=original_shape,
            patch_size=patch_size,
            step_size=predictor.tile_step_size
        )
        
        # Restore original method
        predictor._internal_maybe_mirror_and_predict = original_predict_method
        extractor.remove_hooks()
        
        return full_volume_features
        
    except Exception as e:
        print(f"Error during full volume feature extraction: {e}")
        predictor._internal_maybe_mirror_and_predict = original_predict_method
        extractor.remove_hooks()
        raise e


def demo_full_volume_extraction():
    """
    Extract features for all cases in the dataset.
    
    This function extracts features from THREE specific layers:
    1. decoder.stages.5.convs.0.conv → feature_neg2 (second-to-last conv before batch norm/activation)
    2. decoder.stages.5.convs.1.conv → feature_neg1 (last conv before logits, before batch norm/activation)
    3. decoder.seg_layers.5 → logits (segmentation head output)
    
    All three feature sets are saved in the same .pt file for each case.
    """
    
    model_folder = "/home/cynthia0429/projects/nnUNet/nnUNet_results/Dataset201_BxMR_withRegions/nnUNetTrainer__nnUNetPlans__3d_fullres"
    input_folder = "/data/cynthia0429/nnUNet_raw/Dataset201_BxMR_withRegions_3SEQ/imagesTr"
    output_folder = "/data/cynthia0429/Ensemble_project/MTKD-RL/nnUNet_201_features_3_layers"
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all cases
    from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
    import json
    
    with open(os.path.join(model_folder, 'dataset.json'), 'r') as f:
        dataset_json = json.load(f)
    
    file_lists = create_lists_from_splitted_dataset_folder(input_folder, dataset_json['file_ending'])
    
    if not file_lists:
        print("No files found!")
        return
    
    print(f"Found {len(file_lists)} cases to process")
    print(f"Output folder: {output_folder}")
    print("="*60)
    
    processed = 0
    failed = 0
    
    for i, case_files in enumerate(file_lists):
        case_name = os.path.basename(case_files[0]).split('_0000')[0]
        
        print(f"\n[{i+1}/{len(file_lists)}] Processing case: {case_name}")
        print(f"Files: {[os.path.basename(f) for f in case_files]}")
        
        try:
            # Extract features for the FULL volume
            full_features = extract_full_volume_features(
                model_folder=model_folder,
                input_files=case_files,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            if full_features:
                # Organize features with the requested naming scheme
                # Convert to bfloat16 to save space while maintaining numerical stability
                organized_features = {}
                for layer_name, features in full_features.items():
                    # Convert to bfloat16 (50% space savings vs fp32)
                    features_bf16 = features.to(torch.bfloat16)
                    
                    if layer_name == 'feature_neg2':
                        organized_features['feature_neg2'] = features_bf16
                        print(f"  ✅ feature_neg2 (conv0): {features_bf16.shape} [bf16]")
                    elif layer_name == 'feature_neg1':
                        organized_features['feature_neg1'] = features_bf16
                        print(f"  ✅ feature_neg1 (conv1): {features_bf16.shape} [bf16]")
                    elif layer_name == 'logits':
                        organized_features['logits'] = features_bf16
                        print(f"  ✅ logits (seg_layers): {features_bf16.shape} [bf16]")
                    else:
                        organized_features[layer_name] = features_bf16
                        print(f"  ✅ {layer_name}: {features_bf16.shape} [bf16]")
                
                # Save all three feature sets in the same .pt file
                output_file = os.path.join(output_folder, f"features_{case_name}.pt")
                torch.save(organized_features, output_file)
                
                print(f"  💾 Saved all features to: {os.path.basename(output_file)}")
                print(f"      Keys in file: {list(organized_features.keys())}")
                processed += 1
                
            else:
                print(f"  ❌ No features were extracted for {case_name}")
                failed += 1
            
        except Exception as e:
            print(f"  ❌ Error processing {case_name}: {e}")
            failed += 1
            continue
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"✅ Successfully processed: {processed}/{len(file_lists)} cases")
    print(f"❌ Failed: {failed}/{len(file_lists)} cases")
    print(f"📁 Features saved in: {output_folder}")
    print("="*60)


if __name__ == "__main__":
    # Run full extraction for all cases
    demo_full_volume_extraction()
