import torch
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


def normalize_weights(weights: np.ndarray, normalization_type: str) -> np.ndarray:
    if normalization_type == "token-wise":
        if len(weights.shape) != 2 or weights.shape[1] < 2:
            print("Cannot apply token-wise normalization to one sentence, setting global normalization")
            normalization_type = "global"
        else:
            for tok_idx in range(weights.shape[1]):
                max_, min_ = weights[:, tok_idx].max(), weights[:, tok_idx].min()
                weights[:, tok_idx] = (weights[:, tok_idx] - min_) / (max_ - min_)
            normalized_weights = weights
    if normalization_type == "sentence-wise":
        if len(weights[0]) == 1: 
            print("Cannot apply sentence-wise normalization to one word, setting global normalization")
            normalization_type = "global"
        else:
            normalized_weights = []
            for layer_idx in range(len(weights)):
                max_, min_ = weights[layer_idx].max(), weights[layer_idx].min()
                normalized_weights.append((weights[layer_idx] - min_) / (max_ - min_))
            normalized_weights = np.array(normalized_weights) 
    if normalization_type == "global":
        max_, min_ = weights.max(), weights.min()
        normalized_weights = (weights - min_) / (max_ - min_)
    return normalized_weights

def replace_bad_chars(input_string):
    encoded_string = input_string.encode('windows-1251', errors="replace")
    decoded_string = encoded_string.decode('windows-1251')
    return decoded_string

def logit_lens(model, tokenizer, text):
    device = model.device
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    labels = input_ids.clone()
    labels[:, 0] = -100
    target_ids = labels[:, 1:].contiguous()

    with torch.no_grad():
        outputs = model(input_ids, labels=labels, output_hidden_states=True)
        hiddens = outputs.hidden_states

        losses = []
        predictions = []
        decoded_words = []
        for layer_idx in range(len(hiddens)):
            y = hiddens[layer_idx]

            # assert model.config.pretraining_tp == 1 # см код modeling_llama.py, если это не выполнено, там более хитро считаются логиты
            logits = model.lm_head(y).float()
            ids = logits[:, :-1, :].argmax(dim=-1)
            predictions.append(ids[0])
            softmax_logits = torch.softmax(logits, dim=-1)[:, :-1, :]
            # print(softmax_logits.shape)
            per_token_loss = -torch.log(softmax_logits[range(softmax_logits.shape[0]), range(softmax_logits.shape[1]), ids])
            losses.append(per_token_loss[0])
            decoded_words.append([replace_bad_chars(tokenizer.decode([tok])) for tok in ids[0]])

    # num_tokens = len(decoded_words[0])
    # for j in range(num_tokens):
    #     max_len = max([len(decoded_words[i][j]) for i in range(len(decoded_words))])
    #     for i in range(len(decoded_words)):
    #         l = len(decoded_words[i][j])
    #         if l > 0:
    #             decoded_words[i][j] += (max_len - l) * ' '
    losses = torch.stack(losses).cpu().detach().numpy()
    predictions = torch.stack(predictions)

    return predictions, losses, decoded_words


def get_text_size(text, font_size=12):
    fig, ax = plt.subplots()
    text_artist = ax.text(0, 0, text, fontsize=font_size)
    bbox = text_artist.get_window_extent(renderer=fig.canvas.get_renderer())
    plt.close(fig)
    return bbox.width, bbox.height

def plot_word_table(words, weights, top_labels):
    nrows, ncols = len(words), len(words[0])

    # Определим ширину каждой колонки и высоту строк, увеличенные в 1.5 раза
    col_widths = [1.5 * max(get_text_size(words[row][col])[0] for row in range(nrows)) for col in range(ncols)]
    row_heights = [1.5 * max(get_text_size(words[row][col])[1] for col in range(ncols)) for row in range(nrows)]
    label_heights = [1.5 * get_text_size(label)[1] for label in top_labels]

    # Коэффициенты для перевода из пикселей в фигуру
    fig_width = sum(col_widths) / plt.gcf().dpi
    fig_height = (sum(row_heights) + label_heights[0]) / plt.gcf().dpi

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Получаем цветовую карту
    ver = mpl.__version__
    if ver < '3.9.0':
        cmap = mpl.cm.get_cmap('RdYlGn')
    else:
        cmap = plt.get_cmap('RdYlGn')

    # Отрисовка слов в ячейках с соответствующими размерами
    for i in range(nrows):
        for j in range(ncols):
            weight = weights[i][j]
            color = cmap(weight)
            color_with_alpha = (color[0], color[1], color[2], 0.5)  # Устанавливаем прозрачность в 0.5
            rect = plt.Rectangle((sum(col_widths[:j]) / plt.gcf().dpi, sum(row_heights[:i]) / plt.gcf().dpi + label_heights[0] / plt.gcf().dpi),
                                 col_widths[j] / plt.gcf().dpi, row_heights[i] / plt.gcf().dpi, facecolor=color_with_alpha)
            ax.add_patch(rect)
            plt.text(sum(col_widths[:j]) / plt.gcf().dpi + col_widths[j] / (2 * plt.gcf().dpi),
                     sum(row_heights[:i]) / plt.gcf().dpi + row_heights[i] / (2 * plt.gcf().dpi) + label_heights[0] / plt.gcf().dpi,
                     words[i][j], ha='center', va='center', fontsize=12, color='black')

    # Отрисовка меток на белом фоне
    for j in range(ncols):
        rect = plt.Rectangle((sum(col_widths[:j]) / plt.gcf().dpi, 0),
                             col_widths[j] / plt.gcf().dpi, label_heights[0] / plt.gcf().dpi, facecolor='white')
        ax.add_patch(rect)
        plt.text(sum(col_widths[:j]) / plt.gcf().dpi + col_widths[j] / (2 * plt.gcf().dpi),
                 label_heights[0] / (2 * plt.gcf().dpi),
                 top_labels[j], ha='center', va='center', fontsize=12, color='black')

    plt.xlim(0, fig_width)
    plt.ylim(0, fig_height)
    plt.gca().invert_yaxis()
    ax.set_aspect('auto')
    # plt.axis('off')
    fig.show()