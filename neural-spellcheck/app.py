import re

import streamlit as st
import matplotlib.colors as colors
import matplotlib.pyplot as plt

import torch
import transformers as hf # huggingface

"""
# Neural spellcheck
"""

st.image('bert.jpg')

"""
Experimenting with word probability under BERT as a means of "spellcheck".
In fact, a sub-token model is probably needed for spellcheck, since BERT doesn't
know what to do with out-of-vocabulary words.
"""

@st.cache(allow_output_mutation=True)
def load_model():
    model_class = hf.BertForMaskedLM
    tokenizer_class = hf.BertTokenizer
    pretrained_weights = 'bert-base-uncased'
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)
    return model, tokenizer

softmax = torch.nn.Softmax(dim=0)

# this will be slow the first time it runs - we need to download the weights
model, tokenizer = load_model()

input_text = st.text_input(
    label="Enter some text",
    value="Here is some input text."
)

tokens = re.findall(r"[\w']+|[.,!?;]| ", input_text)
token_probabilities = []

for i, token in enumerate(tokens):
    masked_text = tokens[:] # copy by value
    masked_text[i] = tokenizer.mask_token
    masked_text = "".join(masked_text)
    input_ids = torch.tensor([
        tokenizer.encode(
            masked_text,
            add_special_tokens=True
        )
    ])
    masked_position = (
        input_ids.squeeze(0) == tokenizer.mask_token_id
    ).nonzero().item()

    with torch.no_grad():
        output = model(input_ids)

    last_hidden_states = output[0].squeeze(0)
    masked_hidden_state = last_hidden_states[masked_position]

    probabilities = softmax(masked_hidden_state)

    token_id = tokenizer.convert_tokens_to_ids(token)
    token_prob = probabilities[token_id].item()
    token_probabilities.append(token_prob)

cmap = plt.cm.rainbow.reversed()
norm = colors.DivergingNorm(vmin=0, vmax=1, vcenter=0.001)

token_colors = [colors.rgb2hex(cmap(norm(tp))) for tp in token_probabilities]

# this feels like a hack
html_list = [
    '<span style="color: {};"> {} </span>'.format(p, t)
    for p, t in
    zip(token_colors, tokens)
]

"""
Here is your sentence, where the words are coloured by their probability under
BERT, given the other words in the sentence.
Blue means very likely, green is less likely, yellow less still and red
extremely unlikely (following a diverging norm mapping of the matplotlib
rainbow colors).
"""

st.markdown(
    "".join(html_list),
    unsafe_allow_html=True
)

"""
This doesn't work as well as I'd hoped.
Part of the problem is that if one introduces a weird word, that word is used
to compute the probabilities of the other words, which also go low.
It seems to work better with longer sentences.
"""