from os.path import join
from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *


def get_config(runner) -> SemanticSegmentationConfig:
    output_root_uri = '/rastervision/output/mineral_hs'
    # 调换顺序：background=0，mineral=1，匹配你geojson里class_id=1矿化区域
    class_config = ClassConfig(
        names=['background', 'mineral'], colors=['black', 'red'])

    data_base = "/rastervision/data"
    train_image_uri = join(data_base, 'image/nalan.tif')
    train_label_uri = join(data_base, 'label/nalan.geojson')
    val_image_uri = join(data_base, 'image/nalan.tif')
    val_label_uri = join(data_base, 'label/nalan.geojson')

    train_scene = make_scene('scene_train', train_image_uri, train_label_uri, class_config)
    val_scene = make_scene('scene_val', val_image_uri, val_label_uri, class_config)
    scene_dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=[train_scene],
        validation_scenes=[val_scene]
    )

    chip_sz = 128
    # 瓦片采样配置，供给analyze、chip阶段读取
    chip_options = SemanticSegmentationChipOptions(
        sampling=WindowSamplingConfig(
            method=WindowSamplingMethod.random,
            size=chip_sz,
            max_windows=200
        )
    )

    backend = PyTorchSemanticSegmentationConfig(
        data=SemanticSegmentationGeoDataConfig(
            scene_dataset=scene_dataset,
            num_workers=0,
            # 显式地将采样选项映射到你的场景 ID 上
            sampling={
                'scene_train': chip_options.sampling,
                'scene_val': chip_options.sampling
            }
        ),
        model=SemanticSegmentationModelConfig(
            backbone=Backbone.resnet50
        ),
        solver=SolverConfig(lr=1e-4, num_epochs=5, batch_sz=4)
    )

    '''backend = PyTorchSemanticSegmentationConfig(
        data=SemanticSegmentationGeoDataConfig(
            scene_dataset=scene_dataset
        ),
        model=SemanticSegmentationModelConfig(
            backbone=Backbone.resnet50
        ),
        solver=SolverConfig(lr=1e-4, num_epochs=5, batch_sz=1)
    )'''

    return SemanticSegmentationConfig(
        root_uri=output_root_uri,
        dataset=scene_dataset,
        backend=backend,
        chip_options=chip_options,
        predict_options=SemanticSegmentationPredictOptions(chip_sz=chip_sz)
    )


def make_scene(scene_id: str, image_uri: str, label_uri: str,
               class_config: ClassConfig) -> SceneConfig:
    # 读取全部230个高光谱波段
    raster_source = RasterioSourceConfig(
        uris=image_uri,
        channel_order=list(range(230))
    )

    vector_source = GeoJSONVectorSourceConfig(
        uris=label_uri,
        transformers=[
            ClassInferenceTransformerConfig(
                # 所有矢量多边形自动赋值 mineral=1，匹配你geojson class_id=1
                default_class_id=class_config.get_class_id('mineral')
            )
        ])

    label_source = SemanticSegmentationLabelSourceConfig(
        raster_source=RasterizedSourceConfig(
            vector_source=vector_source,
            rasterizer_config=RasterizerConfig(
                # 无矢量空白区域为 background=0
                background_class_id=class_config.get_class_id('background')
            )
        )
    )

    return SceneConfig(
        id=scene_id,
        raster_source=raster_source,
        label_source=label_source,
    )