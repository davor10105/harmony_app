import torch
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision import tv_tensors
import os
from PIL import Image
import xml.etree.ElementTree as ET
import os
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import json
import cv2

imagenet_classes = []
with open('/storage/imagenet_dataset/LOC_synset_mapping.txt', 'r') as f:
    for line in f.readlines():
        label = ' '.join(line.split(' ')[1:])
        imagenet_classes.append(label.split(',')[0].strip())

wmid_to_class = {}
with open('/storage/imagenet_dataset/LOC_synset_mapping.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        wmid = line.split(' ')[0]
        wmid_to_class[wmid] = i




class ClassicImageNet(Dataset):
    def __init__(self, imagenet_path, split, transform):
        assert split in ['train']
        self.dataset = torchvision.datasets.ImageFolder(os.path.join(imagenet_path, 'Data', 'CLS-LOC', split))
        self.transform = transform
    
    def __getitem__(self, i):
        image, label = self.dataset[i]
        image = image.convert('RGB')
        image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.dataset)


class ClassicImageNetValidation(Dataset):
    def __init__(self, imagenet_path, split, transform):
        assert split in ['val', 'test']
        self.image_paths = []
        self.image_names = []
        self.label_paths = []
        for image_name in os.listdir(os.path.join(imagenet_path, 'Data', 'CLS-LOC', split)):
            image_path = os.path.join(imagenet_path, 'Data', 'CLS-LOC', split, image_name)
            self.image_paths.append(image_path)
            self.image_names.append(image_name)
            label_name = image_name.split('.')[0] + '.xml'
            label_path = os.path.join(imagenet_path, 'Annotations', 'CLS-LOC', split, label_name)
            self.label_paths.append(label_path)
        self.transform = transform
    
    def __getitem__(self, i):
        image_path, label_path = self.image_paths[i], self.label_paths[i]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transform(image)
        
        pascalvoc_label = ET.parse(label_path)
        for det_object in pascalvoc_label.iter('object'):
            label = det_object.find('name').text
            break
        label = wmid_to_class[label]
        
        return image, label
    
    def __len__(self):
        return len(self.image_paths)


def get_mask_of_class(mask, v):
    """
    Get binary mask of v-th class.
    :param mask (numpy array, uint8): semantic segmentation mask
    :param v (int): the index of given class
    :return: binary mask of v-th class
    """
    mask_v = (mask == v) * 255
    return mask_v.astype(np.uint8)


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


class SegmentationDataset(Dataset):
    def __init__(self, dataset_path, original_path, transform):
        super().__init__()
        
        original_to_mask = {}
        original_to_path = {}
        original_to_class = {}
        name_to_class = {}
        for class_dir in os.listdir(original_path):
            class_path = os.path.join(original_path, class_dir)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    image_name = image_name.split('.')[0]
                    original_to_mask[image_name] = None
                    original_to_path[image_name] = image_path
                    #original_to_class[image_name] = class_dir
            else:
                image_path = os.path.join(original_path, class_dir)
                image_name = class_dir.split('.')[0]
                original_to_mask[image_name] = None
                original_to_path[image_name] = image_path
                #original_to_class[image_name] = class_dir
                
        #print(len(original_to_path))

        for class_dir in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_dir)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                image_name = image_name.split('.')[0]
                if image_name in original_to_mask:
                    original_to_mask[image_name] = image_path
                    name_to_class[image_name] = class_dir
        
        filtered_original_to_mask = {}
        self.image_names = []
        for image_name, mask_path in original_to_mask.items():
            if mask_path is not None:
                filtered_original_to_mask[image_name] = mask_path
                self.image_names.append(image_name)
        self.original_to_mask = filtered_original_to_mask
        self.original_to_path = original_to_path
        self.original_to_class = original_to_class
        self.name_to_class = name_to_class
        self.transform = transform
        
        self.class_names = []
        with open('/storage-ssd/IMAGENET1K/ImageNet-S/data/categories/ImageNetS_categories_im919.txt', 'r') as f:
            for class_name in f.readlines():
                self.class_names.append(class_name.strip())
            
        self.classes_50 = "goldfish, tiger shark, goldfinch, tree frog, kuvasz, red fox, siamese cat, american black bear, ladybug, sulphur butterfly, wood rabbit, hamster, wild boar, gibbon, african elephant, giant panda, airliner, ashcan, ballpoint, beach wagon, boathouse, bullet train, cellular telephone, chest, clog, container ship, digital watch, dining table, golf ball, grand piano, iron, lab coat, mixing bowl, motor scooter, padlock, park bench, purse, streetcar, table lamp, television, toilet seat, umbrella, vase, water bottle, water tower, yawl, street sign, lemon, carbonara, agaric"
        self.classes_300 = "tench, goldfish, tiger shark, hammerhead, electric ray, ostrich, goldfinch, house finch, indigo bunting, kite, common newt, axolotl, tree frog, tailed frog, mud turtle, banded gecko, american chameleon, whiptail, african chameleon, komodo dragon, american alligator, triceratops, thunder snake, ringneck snake, king snake, rock python, horned viper, harvestman, scorpion, garden spider, tick, african grey, lorikeet, red-breasted merganser, wallaby, koala, jellyfish, sea anemone, conch, fiddler crab, american lobster, spiny lobster, isopod, bittern, crane, limpkin, bustard, albatross, toy terrier, afghan hound, bluetick, borzoi, irish wolfhound, whippet, ibizan hound, staffordshire bullterrier, border terrier, yorkshire terrier, lakeland terrier, giant schnauzer, standard schnauzer, scotch terrier, lhasa, english setter, clumber, english springer, welsh springer spaniel, kuvasz, kelpie, doberman, miniature pinscher, malamute, pug, leonberg, great pyrenees, samoyed, brabancon griffon, cardigan, coyote, red fox, kit fox, grey fox, persian cat, siamese cat, cougar, lynx, tiger, american black bear, sloth bear, ladybug, leaf beetle, weevil, bee, cicada, leafhopper, damselfly, ringlet, cabbage butterfly, sulphur butterfly, sea cucumber, wood rabbit, hare, hamster, wild boar, hippopotamus, bighorn, ibex, badger, three-toed sloth, orangutan, gibbon, colobus, spider monkey, squirrel monkey, madagascar cat, indian elephant, african elephant, giant panda, barracouta, eel, coho, academic gown, accordion, airliner, ambulance, analog clock, ashcan, backpack, balloon, ballpoint, barbell, barn, bassoon, bath towel, beach wagon, bicycle-built-for-two, binoculars, boathouse, bonnet, bookcase, bow, brass, breastplate, bullet train, cannon, can opener, carpenter's kit, cassette, cellular telephone, chain saw, chest, china cabinet, clog, combination lock, container ship, corkscrew, crate, crock pot, digital watch, dining table, dishwasher, doormat, dutch oven, electric fan, electric locomotive, envelope, file, folding chair, football helmet, freight car, french horn, fur coat, garbage truck, goblet, golf ball, grand piano, half track, hamper, hard disc, harmonica, harvester, hook, horizontal bar, horse cart, iron, jack-o'-lantern, lab coat, ladle, letter opener, liner, mailbox, megalith, military uniform, milk can, mixing bowl, monastery, mortar, mosquito net, motor scooter, mountain bike, mountain tent, mousetrap, necklace, nipple, ocarina, padlock, palace, parallel bars, park bench, pedestal, pencil sharpener, pickelhaube, pillow, planetarium, plastic bag, polaroid camera, pole, pot, purse, quilt, radiator, radio, radio telescope, rain barrel, reflex camera, refrigerator, rifle, rocking chair, rubber eraser, rule, running shoe, sewing machine, shield, shoji, ski, ski mask, slot, soap dispenser, soccer ball, sock, soup bowl, space heater, spider web, spindle, sports car, steel arch bridge, stethoscope, streetcar, submarine, swimming trunks, syringe, table lamp, tank, teddy, television, throne, tile roof, toilet seat, trench coat, trimaran, typewriter keyboard, umbrella, vase, volleyball, wardrobe, warplane, washer, water bottle, water tower, whiskey jug, wig, wine bottle, wok, wreck, yawl, yurt, street sign, traffic light, consomme, ice cream, bagel, cheeseburger, hotdog, mashed potato, spaghetti squash, bell pepper, cardoon, granny smith, strawberry, lemon, carbonara, burrito, cup, coral reef, yellow lady's slipper, buckeye, agaric, gyromitra, earthstar, bolete"
        self.classes_919 = "tench, goldfish, great white shark, tiger shark, hammerhead, electric ray, stingray, cock, hen, ostrich, brambling, goldfinch, house finch, junco, indigo bunting, robin, bulbul, jay, magpie, chickadee, water ouzel, kite, bald eagle, vulture, great grey owl, European fire salamander, common newt, eft, spotted salamander, axolotl, bullfrog, tree frog, tailed frog, loggerhead, leatherback turtle, mud turtle, terrapin, box turtle, banded gecko, common iguana, American chameleon, whiptail, agama, frilled lizard, alligator lizard, Gila monster, green lizard, African chameleon, Komodo dragon, African crocodile, American alligator, triceratops, thunder snake, ringneck snake, hognose snake, green snake, king snake, garter snake, water snake, vine snake, night snake, boa constrictor, rock python, Indian cobra, green mamba, sea snake, horned viper, diamondback, sidewinder, trilobite, harvestman, scorpion, black and gold garden spider, barn spider, garden spider, black widow, tarantula, wolf spider, tick, centipede, black grouse, ptarmigan, ruffed grouse, prairie chicken, peacock, quail, partridge, African grey, macaw, sulphur-crested cockatoo, lorikeet, coucal, bee eater, hornbill, hummingbird, jacamar, toucan, drake, red-breasted merganser, goose, black swan, tusker, echidna, platypus, wallaby, koala, wombat, jellyfish, sea anemone, brain coral, flatworm, nematode, conch, snail, slug, sea slug, chiton, chambered nautilus, Dungeness crab, rock crab, fiddler crab, king crab, American lobster, spiny lobster, crayfish, hermit crab, isopod, white stork, black stork, spoonbill, flamingo, little blue heron, American egret, bittern, crane, limpkin, European gallinule, American coot, bustard, ruddy turnstone, red-backed sandpiper, redshank, dowitcher, oystercatcher, pelican, king penguin, albatross, grey whale, killer whale, dugong, sea lion, Chihuahua, Japanese spaniel, Maltese dog, Pekinese, Shih-Tzu, Blenheim spaniel, papillon, toy terrier, Rhodesian ridgeback, Afghan hound, basset, beagle, bloodhound, bluetick, black-and-tan coonhound, Walker hound, English foxhound, redbone, borzoi, Irish wolfhound, Italian greyhound, whippet, Ibizan hound, Norwegian elkhound, otterhound, Saluki, Scottish deerhound, Weimaraner, Staffordshire bullterrier, American Staffordshire terrier, Bedlington terrier, Border terrier, Kerry blue terrier, Irish terrier, Norfolk terrier, Norwich terrier, Yorkshire terrier, wire-haired fox terrier, Lakeland terrier, Sealyham terrier, Airedale, cairn, Australian terrier, Dandie Dinmont, Boston bull, miniature schnauzer, giant schnauzer, standard schnauzer, Scotch terrier, Tibetan terrier, silky terrier, soft-coated wheaten terrier, West Highland white terrier, Lhasa, flat-coated retriever, curly-coated retriever, golden retriever, Labrador retriever, Chesapeake Bay retriever, German short-haired pointer, vizsla, English setter, Irish setter, Gordon setter, Brittany spaniel, clumber, English springer, Welsh springer spaniel, cocker spaniel, Sussex spaniel, Irish water spaniel, kuvasz, schipperke, groenendael, malinois, briard, kelpie, komondor, Old English sheepdog, Shetland sheepdog, collie, Border collie, Bouvier des Flandres, Rottweiler, German shepherd, Doberman, miniature pinscher, Greater Swiss Mountain dog, Bernese mountain dog, Appenzeller, EntleBucher, boxer, bull mastiff, Tibetan mastiff, French bulldog, Great Dane, Saint Bernard, Eskimo dog, malamute, Siberian husky, dalmatian, affenpinscher, basenji, pug, Leonberg, Newfoundland, Great Pyrenees, Samoyed, Pomeranian, chow, keeshond, Brabancon griffon, Pembroke, Cardigan, toy poodle, miniature poodle, standard poodle, Mexican hairless, timber wolf, white wolf, red wolf, coyote, dingo, dhole, African hunting dog, hyena, red fox, kit fox, Arctic fox, grey fox, tabby, tiger cat, Persian cat, Siamese cat, Egyptian cat, cougar, lynx, leopard, snow leopard, jaguar, lion, tiger, cheetah, brown bear, American black bear, ice bear, sloth bear, mongoose, meerkat, tiger beetle, ladybug, ground beetle, long-horned beetle, leaf beetle, dung beetle, rhinoceros beetle, weevil, fly, bee, ant, grasshopper, cricket, walking stick, cockroach, mantis, cicada, leafhopper, lacewing, dragonfly, damselfly, admiral, ringlet, monarch, cabbage butterfly, sulphur butterfly, lycaenid, starfish, sea urchin, sea cucumber, wood rabbit, hare, Angora, hamster, porcupine, fox squirrel, marmot, beaver, guinea pig, sorrel, zebra, hog, wild boar, warthog, hippopotamus, ox, water buffalo, bison, ram, bighorn, ibex, hartebeest, impala, gazelle, Arabian camel, llama, weasel, mink, polecat, black-footed ferret, otter, skunk, badger, armadillo, three-toed sloth, orangutan, gorilla, chimpanzee, gibbon, siamang, guenon, patas, baboon, macaque, langur, colobus, proboscis monkey, marmoset, capuchin, howler monkey, titi, spider monkey, squirrel monkey, Madagascar cat, indri, Indian elephant, African elephant, lesser panda, giant panda, barracouta, eel, coho, rock beauty, anemone fish, sturgeon, gar, lionfish, puffer, abacus, abaya, academic gown, accordion, acoustic guitar, aircraft carrier, airliner, airship, ambulance, amphibian, analog clock, apiary, apron, ashcan, assault rifle, backpack, balloon, ballpoint, Band Aid, banjo, barbell, barber chair, barn, barometer, barrel, barrow, baseball, basketball, bassinet, bassoon, bath towel, bathtub, beach wagon, beacon, beaker, bearskin, beer bottle, beer glass, bib, bicycle-built-for-two, binder, binoculars, birdhouse, boathouse, bobsled, bolo tie, bonnet, bookcase, bow, bow tie, brass, brassiere, breastplate, broom, bucket, buckle, bulletproof vest, bullet train, cab, caldron, candle, cannon, canoe, can opener, cardigan, car mirror, carpenter's kit, carton, cassette, cassette player, castle, catamaran, cello, cellular telephone, chain, chainlink fence, chain saw, chest, chiffonier, chime, china cabinet, Christmas stocking, church, cleaver, cloak, clog, cocktail shaker, coffee mug, coffeepot, combination lock, container ship, convertible, corkscrew, cornet, cowboy boot, cowboy hat, cradle, crane, crash helmet, crate, crib, Crock Pot, croquet ball, crutch, cuirass, desk, dial telephone, diaper, digital clock, digital watch, dining table, dishrag, dishwasher, doormat, drilling platform, drum, drumstick, dumbbell, Dutch oven, electric fan, electric guitar, electric locomotive, envelope, espresso maker, face powder, feather boa, file, fireboat, fire engine, fire screen, flagpole, flute, folding chair, football helmet, forklift, fountain pen, four-poster, freight car, French horn, frying pan, fur coat, garbage truck, gasmask, gas pump, goblet, go-kart, golf ball, golfcart, gondola, gong, gown, grand piano, guillotine, hair slide, hair spray, half track, hammer, hamper, hand blower, hand-held computer, handkerchief, hard disc, harmonica, harp, harvester, hatchet, holster, honeycomb, hook, hoopskirt, horizontal bar, horse cart, hourglass, iPod, iron, jack-o'-lantern, jean, jeep, jersey, jigsaw puzzle, jinrikisha, joystick, kimono, knee pad, knot, lab coat, ladle, lawn mower, lens cap, letter opener, lifeboat, lighter, limousine, liner, lipstick, Loafer, lotion, loudspeaker, loupe, magnetic compass, mailbag, mailbox, manhole cover, maraca, marimba, mask, matchstick, maypole, measuring cup, medicine chest, megalith, microphone, microwave, military uniform, milk can, minibus, miniskirt, minivan, missile, mitten, mixing bowl, mobile home, Model T, modem, monastery, monitor, moped, mortar, mortarboard, mosque, mosquito net, motor scooter, mountain bike, mountain tent, mouse, mousetrap, moving van, muzzle, nail, neck brace, necklace, nipple, notebook, obelisk, oboe, ocarina, odometer, oil filter, organ, oscilloscope, overskirt, oxcart, oxygen mask, paddle, paddlewheel, padlock, paintbrush, pajama, palace, panpipe, parachute, parallel bars, park bench, parking meter, passenger car, pay-phone, pedestal, pencil box, pencil sharpener, perfume, Petri dish, photocopier, pick, pickelhaube, picket fence, pickup, pier, piggy bank, pill bottle, pillow, ping-pong ball, pinwheel, pirate, pitcher, plane, planetarium, plastic bag, plate rack, plunger, Polaroid camera, pole, police van, poncho, pool table, pop bottle, pot, potter's wheel, power drill, prayer rug, printer, prison, projector, puck, punching bag, purse, quill, quilt, racer, racket, radiator, radio, radio telescope, rain barrel, recreational vehicle, reel, reflex camera, refrigerator, remote control, revolver, rifle, rocking chair, rubber eraser, rugby ball, rule, running shoe, safe, safety pin, saltshaker, sandal, sarong, sax, scabbard, scale, school bus, schooner, scoreboard, screw, screwdriver, seat belt, sewing machine, shield, shoji, shopping basket, shopping cart, shovel, shower cap, shower curtain, ski, ski mask, sleeping bag, slide rule, slot, snowmobile, snowplow, soap dispenser, soccer ball, sock, solar dish, sombrero, soup bowl, space heater, space shuttle, spatula, speedboat, spider web, spindle, sports car, spotlight, steam locomotive, steel arch bridge, steel drum, stethoscope, stole, stone wall, stopwatch, stove, strainer, streetcar, stretcher, studio couch, stupa, submarine, suit, sundial, sunglass, suspension bridge, swab, sweatshirt, swimming trunks, swing, syringe, table lamp, tank, teapot, teddy, television, tennis ball, thatch, theater curtain, thimble, thresher, throne, tile roof, toaster, toilet seat, torch, totem pole, tow truck, tractor, trailer truck, tray, trench coat, tricycle, trimaran, tripod, triumphal arch, trolleybus, trombone, typewriter keyboard, umbrella, unicycle, upright, vacuum, vase, velvet, vending machine, vestment, viaduct, violin, volleyball, waffle iron, wall clock, wallet, wardrobe, warplane, washbasin, washer, water bottle, water jug, water tower, whiskey jug, whistle, wig, window screen, window shade, Windsor tie, wine bottle, wok, wooden spoon, worm fence, wreck, yawl, yurt, comic book, street sign, traffic light, menu, plate, guacamole, consomme, trifle, ice cream, ice lolly, French loaf, bagel, pretzel, cheeseburger, hotdog, mashed potato, head cabbage, broccoli, cauliflower, zucchini, spaghetti squash, acorn squash, butternut squash, cucumber, artichoke, bell pepper, cardoon, mushroom, Granny Smith, strawberry, orange, lemon, fig, pineapple, banana, jackfruit, custard apple, pomegranate, hay, carbonara, dough, meat loaf, pizza, potpie, burrito, cup, eggnog, bubble, cliff, coral reef, ballplayer, scuba diver, rapeseed, daisy, yellow lady's slipper, corn, acorn, hip, buckeye, coral fungus, agaric, gyromitra, stinkhorn, earthstar, hen-of-the-woods, bolete, ear, toilet tissue"
        self.classes_50 = ['background'] + self.classes_50.split(', ')
        self.classes_300 = ['background'] + self.classes_300.split(', ')
        self.classes_919 = ['background'] + self.classes_919.split(', ')

    def __getitem__(self, item):
        image_name = self.image_names[item]
        #class_name = self.original_to_class[image_name]
        class_name = self.name_to_class[image_name]
        #print(class_name)
        #print(class_name)
        image_path = self.original_to_path[image_name]
        mask_path = self.original_to_mask[image_name]
        class_index = self.class_names.index(class_name)
        
        target_name = self.classes_919[class_index + 1]
        imagenet_class_name = imagenet_classes.index(target_name)
        
        image = Image.open(image_path)
        #image = np.uint8(Image.open(image_path))
        #image = torch.from_numpy(image)
        
        
        gt = Image.open(mask_path)
        gt = np.array(gt)
        gt = gt[:, :, 1] * 256 + gt[:, :, 0]

        # Get boundary mask for each class.
        #boundary_gt = self.get_boundary_mask(gt + 1)

        #gt = torch.from_numpy(gt.astype(np.float32))
        #boundary_gt = torch.from_numpy(boundary_gt.astype(np.float32))

        #gt = gt.view(-1)
        #boundary_gt = boundary_gt.view(-1)

        #mask = gt != 1000
        #gt = gt[mask]
        #boundary_gt = boundary_gt[mask]
        #mask = get_mask_of_class(gt + 1, class_index)
        mask = torch.from_numpy(gt == (class_index + 1)).float()
        #mask = (mask == (class_index + 1)).float()
        IW, IH = image.size
        MH, MW = mask.shape
        if IW == MH and IH == MW and IW != IH:
            mask = cv2.rotate(mask.numpy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
            mask = torch.from_numpy(mask)
        #print(item, image.size, mask.shape)
        try:
            image, mask = self.transform(image, tv_tensors.Mask(mask))
        except Exception as e:
            print(item, image.size, mask.shape)
            raise e

        return image, mask, imagenet_class_name
    
    def get_boundary_mask(self, mask):
        boundary = np.zeros_like(mask).astype(mask.dtype)
        for v in np.unique(mask):
            mask_v = get_mask_of_class(mask, v)
            boundary_v = mask_to_boundary(mask_v, dilation_ratio=0.03)
            boundary += (boundary_v > 0) * v
        return boundary

    def __len__(self):
        return len(self.image_names)