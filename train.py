import json
from fastai.vision.all import *

default_device()

data_base = Path(".").resolve()
data_dir = data_base / "train_mini"
data_json = data_base / "train_mini.json"

# Probably only need Arthropoda from Animalia
folders = [*data_dir.glob("*")]
kingdoms = set()
animalia = set()

for folder in tqdm(folders):
    if not folder.is_dir():
        continue
    names = folder.name.split("_")
    kingdom, phylum = names[1:3]
    kingdoms.add(kingdom)
    if kingdom == "Animalia":
        animalia.add(phylum)

def get_annotes(fname):
    annot_dict = json.load(open(fname))
    id2images, id2cats = {}, collections.defaultdict(list)
    classes = {o['id']: o['name'] for o in annot_dict['categories']}
    for o in annot_dict['annotations']:
        id2cats[o['image_id']].append(classes[o['category_id']])
    id2images = {o['id']: data_base / o['file_name'] for o in annot_dict['images']}
    ids = list(id2images.keys())
    return [id2images[k] for k in ids], [id2cats[k] for k in ids]

imgs, labels = get_annotes(data_json)
labels = [label[0] for label in labels] # each label is a one-item list

# Remove non-Arthropoda Animalia from dataset
new_imgs, new_labels = [], []
start_idx = len(str(imgs[0].parents[1])) + 1
for i, (img, label) in enumerate(zip(imgs, labels)):
    tag = str(img.parent)[start_idx:]
    if "Animalia" not in tag:
        new_imgs.append(img)
        new_labels.append(label)
    elif "Arthropoda" in tag:
        new_imgs.append(img)
        new_labels.append(label)
        
imgs, labels = new_imgs, new_labels

dls = ImageDataLoaders.from_lists(data_base, imgs, labels, valid_pct=0.2,
                                   bs=64, item_tfms=Resize(460),
                                   batch_tfms=aug_transforms(size=224))

learn = vision_learner(dls, resnet18, metrics=error_rate)

learn.fit_one_cycle(10, 2e-2)

learn.save("10epochs")

lr = learn.lr_find()
print(lr)

learn.fit_one_cycle(15, 1e-3)

learn.save("25epochs")

lr = learn.lr_find()
print(lr)

learn.fit_one_cycle(15, lr)

learn.save("40epochs")
