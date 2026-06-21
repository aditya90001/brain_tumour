import torch

def predict_image(model, image, transform, device, classes):

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1)

    return classes[pred.item()]