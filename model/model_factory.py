
from model.custom_cnn import CustomCNN
from model.resnet18 import ResNet18
from model.mobilenetv2 import MobileNetV2
from model.alexnet import AlexNet


def get_model(
    model_name,
    num_classes=14,
    pretrained=False
):

    model_name = model_name.lower().strip()

    if model_name == "custom_cnn":

        # Custom CNN does not use pretrained weights
        return CustomCNN(
            num_classes=num_classes
        )

    elif model_name == "resnet18":

        return ResNet18(
            num_classes=num_classes,
            pretrained=pretrained
        )

    elif model_name == "mobilenetv2":

        return MobileNetV2(
            num_classes=num_classes,
            pretrained=pretrained
        )

    elif model_name == "alexnet":

        return AlexNet(
            num_classes=num_classes,
            pretrained=pretrained
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}. "
            "Choose custom_cnn, resnet18, "
            "mobilenetv2, or alexnet."
        )


if __name__ == "__main__":

    models = [
        "custom_cnn",
        "resnet18",
        "mobilenetv2",
        "alexnet"
    ]

    for name in models:

        model = get_model(
            model_name=name,
            num_classes=14,
            pretrained=False
        )

        parameters = sum(
            p.numel() for p in model.parameters()
        )

        print(
            f"{name:15s} | Parameters: {parameters:,}"
        )
