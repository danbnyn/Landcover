from pathlib import Path
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
from tifffile import imread
import collections

class Sentinel2Landcover(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Sentinel-2 landcover data."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial Release',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        # Add class distribution to the dataset description or in metadata
        return tfds.core.DatasetInfo(
            builder=self,
            description="The input data contains 256x256 pixel images of four bands covering the visible spectrum as "
                        "well as near-infrared (R-G-B-NIR). Those images are extracted from larger Sentinel-2 images "
                        "over the European continent. Every pixel covers a 10 meters square area. In addition, "
                        "a segmentation mask is provided for every image, where each pixel contains one land cover "
                        "label, encoded as an integer. The land cover labels are drawn from the ten different "
                        "classes. The land cover data is a simplified version of a subset of the S2GLC data (The "
                        "Sentinel-2 Global Land Cover (S2GLC) project was founded by the European Space Agency). The "
                        "input images are 16-bits numpy files and the land cover masks are 8-bits numpy files.",

            features=tfds.features.FeaturesDict({
                'image': tfds.features.Tensor(shape=(4, 256, 256), dtype=np.uint16),
                'segmentation_mask': tfds.features.Tensor(shape=(256, 256), dtype=np.uint8),
            }),

            supervised_keys=('image', 'segmentation_mask'),
            metadata=tfds.core.MetadataDict({
                'class_distribution': self.class_distribution
            }),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        # Initialize class distribution as a counter
        self.class_distribution = collections.Counter()

        dl_manager.download_and_extract()

        # Set the path to the extracted data
        dataset_path = dl_manager.extract(Path('/data') / 'sentinel2_landcover.zip')

        return {
            'train': self._generate_examples(dataset_path / 'train'),
            'test': self._generate_examples(dataset_path / 'test'),
        }

    def _generate_examples(self, path: Path):
        """Yields examples and updates class distribution."""
        image_dir = path / 'images'
        mask_dir = path / 'masks'

        for image_file in tqdm(sorted(image_dir.glob('*.tif')), desc=f"Processing {path.name} set"):
            image_id = image_file.stem
            mask_file = mask_dir / f"{image_id}.tif"

            if not mask_file.exists():
                print(f"Warning: Mask file not found for {image_id}")
                continue

            try:
                image = imread(str(image_file))
                mask = imread(str(mask_file))

                # Ensure correct shapes and dtypes
                image = image.astype(np.uint16)
                mask = mask.astype(np.uint8)

                # swap axes to match the expected shape from 256, 256, 4 -> 4, 256, 256
                image = np.moveaxis(image, -1, 0)

                if image.shape != (4, 256, 256):
                    print(f"Warning: Unexpected image shape {image.shape} for {image_id}")
                    continue

                if mask.shape != (256, 256):
                    print(f"Warning: Unexpected mask shape {mask.shape} for {image_id}")
                    continue

                # Update class distribution from the mask
                unique, counts = np.unique(mask, return_counts=True)
                self.class_distribution.update(dict(zip(unique, counts)))

                yield image_id, {
                    'image': image,
                    'segmentation_mask': mask,
                }

            except Exception as e:
                print(f"Error processing {image_id}: {str(e)}")