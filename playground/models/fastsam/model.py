from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.engine.exporter import Exporter
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, ROOT, is_git_dir
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.torch_utils import model_info, smart_inference_mode

import sys
sys.path.append('/home/joonmyung/project/joonmyung')
from playground.models.fastsam.predict import FastSAMPredictor
from playground.models.fastsam.prompt import FastSAMPrompt
from joonmyung.meta_data import imnet_label
from PIL import Image
from glob import glob
from tqdm import tqdm
import argparse
import torch
import os

class FastSAM(YOLO):
    @smart_inference_mode()
    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The prediction results.
        """
        if source is None:
            source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        overrides = self.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        self.predictor = FastSAMPredictor(overrides=overrides)
        self.predictor.setup_model(model=self.model, verbose=False)
        try:
            return self.predictor(source, stream=stream)
        except Exception as e:
            return None

    def train(self, **kwargs):
        """Function trains models but raises an error as FastSAM models do not support training."""
        raise NotImplementedError("Currently, the training codes are on the way.")

    def val(self, **kwargs):
        """Run validation given dataset."""
        overrides = dict(task='segment', mode='val')
        overrides.update(kwargs)  # prefer kwargs
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)
        validator = FastSAM(args=args)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    @smart_inference_mode()
    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """
        overrides = dict(task='detect')
        overrides.update(kwargs)
        overrides['mode'] = 'export'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz:
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        if args.batch == DEFAULT_CFG.batch:
            args.batch = 1  # default to 1 if not modified
        return Exporter(overrides=args)(model=self.model)

    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def __getattr__(self, attr):
        """Raises error if object has no requested attribute."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, default=0, help="")
    args = parser.parse_args()

    device, train = "cuda", True
    box_prompt, text_prompt, point_prompt, point_label = None, None, None, None
    model_path, root_path = "/hub_data1/joonmyung/weights/FastSAM/FastSAM-x.pt", '/hub_data1/joonmyung/data/imagenet'
    data_len, data_split_num, split_full = 50, 0, 8

    split = args.split

    model = FastSAM(model_path)
    class_dirs = sorted(os.listdir(os.path.join(root_path, "train")))
    class_split_len = len(class_dirs) // split_full
    class_dirs = class_dirs[split * class_split_len:(split + 1) * class_split_len]
    for c_idx, class_name in enumerate(tqdm(class_dirs)):
        class_dir = os.path.join(root_path, "train", class_name)
        qual_path = os.path.join(root_path, "train_quality", class_name)
        obj_path = os.path.join(root_path, "train_obj", class_name)
        os.makedirs(qual_path, exist_ok=True)
        os.makedirs(obj_path, exist_ok=True)

        data_paths = sorted(glob(os.path.join(class_dir, "*.JPEG")))
        text_prompt = label = imnet_label[c_idx]
        print(f"class_dir : {class_dir}, label : {label}")
        for idx, data_path in enumerate(data_paths[data_len * data_split_num:]):
            if idx == data_len:
                break

            try:
                img = Image.open(data_path).convert("RGB")
                output_path = data_path.replace("train", "train_obj").replace("JPEG", "pt")

                if os.path.isfile(output_path):
                    continue
                everything_results = model(img, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
                if everything_results:
                    prompt_process = FastSAMPrompt(img, everything_results, device=device)
                    if box_prompt:
                        ann = prompt_process.box_prompt(bboxes=box_prompt)
                    elif text_prompt:
                        ann = prompt_process.text_prompt(text=text_prompt)
                    elif point_prompt:
                        ann = prompt_process.point_prompt(points=point_prompt, pointlabel=point_label)
                    else:
                        ann = prompt_process.everything_prompt()
                    if len(ann):
                        prompt_process.plot(
                            annotations=ann,
                            output_path=data_path.replace("train", "train_quality"),
                            bboxes=box_prompt,
                            points=point_prompt,
                            point_label=point_label,
                            withContours=False,
                            better_quality=False,
                        )
                        torch.save(torch.from_numpy(ann), output_path)
            except Exception as e:
                print(e)
                continue



