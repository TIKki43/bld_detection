import os
import xml.etree

import keras.callbacks
from numpy import zeros, asarray

import mrcnn.utils
import mrcnn.config
import mrcnn.model

class KangarooDataset(mrcnn.utils.Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        # Adds information (image ID, image path, and annotation file path) about each image in a dictionary.
        self.add_class("dataset", 1, "empty")
        # self.add_class("dataset", 2, "beton")
        # self.add_class("dataset", 3, "finish")
        self.add_class("dataset", 2, "blocks")
        self.add_class("dataset", 3, "window")
        # self.add_class("dataset", 6, "other")
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'

        for filename in os.listdir(images_dir):
            image_id = filename[:-4]

            if is_train and int(image_id) >= 296:
                continue

            if not is_train and int(image_id) < 296:
                continue

            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
            # print(ann_path)

            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids=[0, 1, 2, 3])

    # Loads the binary masks for an image.
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        # print(info)
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            if (box[4] == 'empty'):
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('empty'))
            # elif (box[4] == 'beton'):
            #     masks[row_s:row_e, col_s:col_e, i] = 2
                # class_ids.append(self.class_names.index('beton'))
            # elif (box[4] == 'finish'):
            #     masks[row_s:row_e, col_s:col_e, i] = 3
            #     class_ids.append(self.class_names.index('finish'))
            # elif (box[4] == 'blocks'):
            #     masks[row_s:row_e, col_s:col_e, i] = 4
            #     class_ids.append(self.class_names.index('blocks'))
            elif (box[4] == 'blocks'):
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('blocks'))
            else:
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('window'))
        return masks, asarray(class_ids, dtype='int32')

    # A helper method to extract the bounding boxes from the annotation file
    def extract_boxes(self, filename):
        tree = xml.etree.ElementTree.parse(filename)

        root = tree.getroot()

        boxes = list()
        for box in root.findall('.//object'):
            # name = box.find('name').text
            # xmin = int(box.find('xmin').text)
            # ymin = int(box.find('ymin').text)
            # xmax = int(box.find('xmax').text)
            # ymax = int(box.find('ymax').text)
            # coors = [xmin, ymin, xmax, ymax]
            # boxes.append(coors)
            name = box.find('name').text
            xmin = int(float(box.find('./bndbox/xmin').text))
            ymin = int(float(box.find('./bndbox/ymin').text))
            xmax = int(float(box.find('./bndbox/xmax').text))
            ymax = int(float(box.find('./bndbox/ymax').text))
            # coors = [xmin, ymin, xmax, ymax, name]
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

class KangarooConfig(mrcnn.config.Config):
    NAME = "kangaroo_cfg"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = 4

    STEPS_PER_EPOCH = 60

# Train
train_dataset = KangarooDataset()
train_dataset.load_dataset(dataset_dir='aug_all', is_train=True)

train_dataset.prepare()

# Validation
validation_dataset = KangarooDataset()
validation_dataset.load_dataset(dataset_dir='aug_all', is_train=True)
validation_dataset.prepare()

# Model Configuration
kangaroo_config = KangarooConfig()

# Build the Mask R-CNN Model Architecture
model = mrcnn.model.MaskRCNN(mode='training', 
                             model_dir='./',
                             config=kangaroo_config)

model.load_weights(filepath='mask_rcnn_coco.h5',
                   by_name=True, 
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# print(train_dataset)
filepath="weights-new4_no_data_improvement-{epoch:02d}.h5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True,verbose=1,
    save_best_only=False, mode='auto', period=1)

model.train(train_dataset=train_dataset,
            val_dataset=validation_dataset,
            learning_rate=kangaroo_config.LEARNING_RATE,
            epochs=30,
            layers='heads')
#
model_path = 'test.h5'
model.keras_model.save_weights(model_path)
