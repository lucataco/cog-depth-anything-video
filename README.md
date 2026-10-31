# Cog depth-anything-video Model

This is an implementation of [JohanDL/Depth-Anything-Video](https://huggingface.co/spaces/JohanDL/Depth-Anything-Video) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own fork of SDXL to [Replicate](https://replicate.com).

## Basic Usage

Run a prediction:

    cog predict -i video=@dolphins.mp4

## Output

![output](output.gif)

## License

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

`depth_anything/` comes from [LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything), which is Apache 2.0.

The model weights are not covered by this license. [LiheYoung/depth-anything-small-hf](https://huggingface.co/LiheYoung/depth-anything-small-hf) is released under the [Apache 2.0](https://github.com/LiheYoung/Depth-Anything/blob/main/LICENSE) license.
