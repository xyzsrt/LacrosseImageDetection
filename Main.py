if __name__ == '__main__':
    import multiprocessing

    multiprocessing.freeze_support()

    import os
    import torch
    from ultralytics import YOLO
    import cv2
    import matplotlib.pyplot as plt


    dataset_location = r"G:\pythonProject\LacrosseVision\LacrosseVision.v7i.yolov8"

    data_yaml_path = os.path.join(dataset_location, 'data.yaml')


    model = YOLO('yolov8s.pt')

    model.train(
        data=data_yaml_path,
        epochs=25,
        imgsz=800,
        plots=True,
    )

    results = model.val(data=data_yaml_path)
    print(results)

    best_model_path = r"G:\pythonProject\LacrosseVision\runs\detect\weights\best.pt"

    model = YOLO(best_model_path)

    image_path = r"G:\pythonProject\LacrosseVision\Testimage.jpg"

    results = model.predict(source=image_path)

    image = cv2.imread(image_path)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.show()
 
