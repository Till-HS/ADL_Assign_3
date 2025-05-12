import json

import torch
from nltk.translate.bleu_score import corpus_bleu
from torchvision.transforms import transforms
from tqdm import tqdm

from datasets import CaptionDataset

# Parameters
data_folder = './data/processed/'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
checkpoint = './BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
word_map_file = './data/processed/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nb_examples = 5


if __name__=='__main__':
    # Load model
    checkpoint = torch.load(checkpoint, weights_only=False)
    rnn = checkpoint['rnn']
    rnn = rnn.to(device)
    rnn.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)
    first_word = torch.LongTensor([word_map['<start>']]).to(device)
    end_token = torch.LongTensor([word_map['<end>']]).to(device)

    # load data
    transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transform),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # lists to store true and generated captions
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(tqdm(loader)):
        # Move to GPU
        image = image.to(device)  # (1, 3, 256, 256)

        # Encode
        encoder_out = encoder.forward(image)  # (1, encoding_pixel_len, encoding_pixel_len, encoder_dim)

        # Create caption
        gen_cap = rnn.test(encoder_out, first_word, end_token)

        # store the true caption and the generated one
        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start>, <end> and <pads>
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in gen_cap if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

    # calculate and print bleu score
    bleu = corpus_bleu(references, hypotheses)
    print(f'bleu score: {bleu}\n')

    # print examples
    print('examples: \n')
    for i in range(nb_examples):
        hyp_example, ref_example = [], []
        for k in range(len(references[i])):
            ref = []
            for j in range(len(references[i][k])):
                ref.append(rev_word_map[references[i][k][j]])
            ref_example.append(ref)
        for j in range(len(hypotheses[i])):
            hyp_example.append(rev_word_map[hypotheses[i][j]])
        print(f'generated caption: {hyp_example}, true caption(s): {ref_example}')
