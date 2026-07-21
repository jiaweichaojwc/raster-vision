from os.path import join
from pathlib import Path  # 引入 pathlib 用于方便地遍历文件夹
from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *


def get_config(runner) -> SemanticSegmentationConfig:
    output_root_uri = '/rastervision/output/mineral_hs'
    class_config = ClassConfig(
        names=['background', 'mineral'], colors=['black', 'red'])

    # 1. 动态读取整个文件夹
    data_base = Path("/rastervision/data")
    image_dir = data_base / "image"
    label_dir = data_base / "label"

    # 找到 image 文件夹下所有的 .tif 文件
    image_paths = sorted(image_dir.glob("*.tif"))

    all_scenes = []

    # 遍历所有图像文件，并寻找同名的 .geojson 标签文件
    for img_path in image_paths:
        # 获取文件名 (不含后缀)，例如 "nalan_01"
        file_stem = img_path.stem
        label_path = label_dir / f"{file_stem}.geojson"

        # 确保对应的标签文件存在
        if label_path.exists():
            # 使用文件名作为 scene_id，保证唯一性
            scene_id = file_stem
            scene = make_scene(
                scene_id,
                str(img_path),
                str(label_path),
                class_config
            )
            all_scenes.append(scene)
        else:
            print(f"⚠️ 警告: 找不到 {img_path.name} 对应的标签文件 {label_path.name}，已跳过。")

    # 2. 划分训练集和验证集 (例如 80% 训练，20% 验证)
    split_idx = int(len(all_scenes) * 0.8)
    train_scenes = all_scenes[:split_idx]
    val_scenes = all_scenes[split_idx:]

    # 如果数据太少，确保验证集至少有一个场景
    if len(val_scenes) == 0 and len(train_scenes) > 0:
        val_scenes = [train_scenes[-1]]

    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=val_scenes
    )

    chip_sz = 128
    chip_options = SemanticSegmentationChipOptions(
        sampling=WindowSamplingConfig(
            method=WindowSamplingMethod.random,
            size=chip_sz,
            max_windows=200
        )
    )

    # 3. 动态生成 sampling 字典，为每一个场景映射采样规则
    # 将所有的 scene_id 绑定到 chip_options.sampling 上
    sampling_dict = {scene.id: chip_options.sampling for scene in all_scenes}

    backend = PyTorchSemanticSegmentationConfig(
        data=SemanticSegmentationGeoDataConfig(
            scene_dataset=scene_dataset,
            num_workers=0,
            sampling=sampling_dict  # <--- 使用动态生成的字典
        ),
        model=SemanticSegmentationModelConfig(
            backbone=Backbone.resnet50
        ),
        # 保持你之前修改成功的 batch_sz=4
        solver=SolverConfig(lr=1e-4, num_epochs=5, batch_sz=4,class_loss_weights=[1.0,50.0,0.0]),
    )

    return SemanticSegmentationConfig(
        root_uri=output_root_uri,
        dataset=scene_dataset,
        backend=backend,
        chip_options=chip_options,
        predict_options=SemanticSegmentationPredictOptions(chip_sz=chip_sz)
    )


def make_scene(scene_id: str, image_uri: str, label_uri: str,
               class_config: ClassConfig) -> SceneConfig:
    raster_source = RasterioSourceConfig(
        uris=image_uri,
        channel_order=list(range(230))
    )

    vector_source = GeoJSONVectorSourceConfig(
        uris=label_uri,
        transformers=[
            ClassInferenceTransformerConfig(
                default_class_id=class_config.get_class_id('mineral')
            )
        ])

    label_source = SemanticSegmentationLabelSourceConfig(
        raster_source=RasterizedSourceConfig(
            vector_source=vector_source,
            rasterizer_config=RasterizerConfig(
                background_class_id=class_config.get_class_id('background')
            )
        )
    )

    return SceneConfig(
        id=scene_id,
        raster_source=raster_source,
        label_source=label_source,
    )