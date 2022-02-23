import pandas as pd
from point_interator import PointIterator
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import os

datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect', cval=125)


def augment(point, cnt):
    filename = "c:\\patches\\" + point.FILE_NAME + "-point3.jpg"
    output_dir = filename + "-aug"
    output_file = output_dir + "\\aug" + str(cnt) + ".jpg"
    image_in = io.imread(filename)
    transformed = datagen.random_transform(image_in)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    io.imsave(output_file, transformed)
    return output_file


csv_file = 'C:/greg/reefscan_ml/from-reefmon/points.csv'
df = pd.read_csv(csv_file)
labels = df.groupby('VIDEO_CODE').size().reset_index().rename(columns={0: 'cnt'})

aug_points = pd.DataFrame()
for index, label in labels.iterrows():
    cnt = label.cnt
    if cnt < 200:
        point_df = df[df.VIDEO_CODE == label.VIDEO_CODE]
        point_interator = PointIterator(point_df)
        while cnt < 200:
            point = point_interator.get()
            output_file = augment(point, cnt)
            cnt += 1
            point_out = point.copy()
            point_out['orig_file'] = point_out.FILE_NAME
            point_out.FILE_NAME = output_file[11:]
            aug_points = aug_points.append(point_out)

aug_points.to_csv(path_or_buf="C:/greg/reefscan_ml/from-reefmon/aug_points.csv")




print(labels)

print(df.shape)