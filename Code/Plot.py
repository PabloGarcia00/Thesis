from datetime import date
from sklearn.metrics import (PrecisionRecallDisplay,
                             RocCurveDisplay,
                             accuracy_score,
                             classification_report,
                             auc,
                             roc_curve,
                             precision_recall_curve,)
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots
# dd/mm/YY H:M:S
# dt_string = now.strftime("%d/%m/%Y %H:%M:%S")


def init_style():

    #mpl.use("pgf")
    plt.style.use("science")

    mpl.rcParams.update({
        "savefig.bbox": "tight",
        "pgf.texsystem": "lualatex",        # Use pdflatex for LaTeX processing
        "text.usetex": True,                # Use LaTeX for text rendering
        # Use serif fonts for consistency with thesis style
        "font.family": "serif",
        "font.size": 8,                    # Base font size
        "axes.titlesize": 1,               # Title font size
        "axes.labelsize": 10,               # Axis labels font size
        "xtick.labelsize": 11,               # X-axis tick label font size
        "ytick.labelsize": 11,               # Y-axis tick label font size
        "legend.fontsize": 11,               # Legend font size
        "lines.linewidth": 1,               # Line width for plot lines
        "lines.markersize": 4,              # Marker size
        "hatch.linewidth": 0.4,
        # Figure size in inches (adjust to fit thesis template)
        "figure.figsize": (2.96, 2.96 * 0.8),
        "figure.dpi": 400,                  # DPI for rasterized output
        "savefig.dpi": 400,                 # DPI for saved figures
        "pgf.preamble": r"""
        \usepackage[utf8]{inputenc}
        \usepackage[T1]{fontenc}
        \usepackage{lmodern}
        \usepackage{siunitx}
        \usepackage{amsmath}
        """,
        "axes.grid": True,                  # Enable grid for readability
        "grid.alpha": 0.3,                  # Grid transparency
        "grid.linestyle": ":",              # Grid line style
    })


init_style()


def to_pgf(filename, *args, **kwargs):
    plt.savefig(filename)


def plot_auroc(y_true, y_pred, name=None):
    assert y_true.shape == y_pred.shape
    y_p = y_pred.cpu().detach().numpy()
    y_t = y_true.cpu().numpy().flatten().astype(int)
    RocCurveDisplay.from_predictions(y_t, y_p)

def plot_auprc(y_true, y_pred):
    # y_p = y_pred[:,1].cpu().detach().numpy()
    # y_p = y_p.squeeze()
    # y_t = y_true[:,1].cpu().numpy().astype(int)
    y_p = y_pred.cpu().detach().numpy()
    y_t = y_true.cpu().flatten().numpy().astype(int)
    PrecisionRecallDisplay.from_predictions(y_t, y_p)
    


def plot_loss(train_loss=None, val_loss=None):
    fig, ax = plt.subplots()
    if train_loss:
        ax.step(range(len(train_loss)), train_loss, label="Training Loss")
    if val_loss:
        ax.step(range(len(val_loss)), val_loss, label="Validation Loss")
    if not train_loss and not val_loss:
        print("NO DATA NO PLOT")
        return None
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig, ax


def accuracy(y_true, y_pred):
    y_p = y_pred.cpu().detach().numpy()
    y_t = y_true.cpu().flatten().numpy().astype(int)

    y_p = (y_p >= 0.5).astype(int)
    score = accuracy_score(y_true=y_t, y_pred=y_p)
    return score


def report(y_true, y_pred):
    y_p = y_pred.cpu().detach().numpy()
    y_t = y_true.cpu().flatten().numpy().astype(int)

    y_p = (y_p >= 0.5).astype(int)
    report = classification_report(y_t, y_p, label=[0, 1])
    return report


def mlc_auroc(y_true, y_pred):
    pred = y_pred.cpu().detach().numpy()
    label = y_true.cpu().detach().numpy()

    fig, ax = plt.subplots()
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")

    for i in range(pred.shape[1]):
        y_p = pred[:, i]
        y_t = label[:, i]
        fpr, tpr, _ = roc_curve(y_t, y_p)
        roc_auc = auc(fpr, tpr)
        ax.step(fpr, tpr, label=f"class_{i} AUC={roc_auc}")
    ax.legend()
    return fig, ax

def mlc_auprc(y_true, y_pred):
    pred = y_pred.cpu().detach().numpy()
    label = y_true.cpu().detach().numpy()

    fig, ax = plt.subplots()
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    for i in range(pred.shape[1]):
        y_p = pred[:, i]
        y_t = label[:, i]
        precision, recall, _ = precision_recall_curve(y_t, y_p)
        prc_auc = auc(recall, precision)
        ax.step(recall, precision, label=f"class_{i} AUC={prc_auc}")
    ax.legend()
    return fig, ax

if __name__ == "__main__":
    import torch
    labels = torch.randint(0, 2, (32, 1))  # Creates 32 rows of either 0 or 1
    labels = torch.cat((labels, 1 - labels), dim=1).to("cuda")
    predictions = torch.rand((32, 2))
    # display_object = plot_auprc(labels, predictions)
    # display_object.plot()
    # plt.show()
    v1, v2 = torch.rand(10), torch.rand(10)
    fig, ax = plot_loss(v1, v2)
    plt.plot()
    plt.show()
