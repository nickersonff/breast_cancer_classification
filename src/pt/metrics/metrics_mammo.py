from logging import info

from matplotlib.pyplot import (
    figure,
    legend,
    plot,
    show,
    title,
    xlabel,
    xlim,
    ylabel,
    ylim,
)
from numpy import array, mean, newaxis
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


class MetricsTracker:
    """
    Classe responsável por acumular predições e calcular todas as métricas clínicas e de validação.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reseta os acumuladores para iniciar uma nova avaliação/época."""
        self.all_labels: list[int] = []
        self.all_preds: list[int] = []
        self.all_probs: list[float] = []
        self.return_probs: list[dict[str, list[float] | int | str]] = []

    def update(
        self,
        images: list[str],
        labels: list[int],
        probs: list[list[float]],
        pred_labels: list[int],
    ):
        """Adiciona os dados de um batch aos acumuladores."""
        for _img_file, _probs, lbl in zip(images, probs, labels):
            p: list[float] = [float(p) for p in _probs]
            self.return_probs.append(
                {
                    "image": str(_img_file),
                    "probs": p,
                    "label": int(lbl.item() if hasattr(lbl, "item") else lbl),
                }
            )
            self.all_probs.append(p[1])  # Probabilidade da classe positiva

        self.all_labels.extend(labels.detach().cpu().numpy())
        self.all_preds.extend(pred_labels.detach().cpu().numpy())

    def compute_metrics(self):
        """Calcula o dicionário completo de métricas com base nos dados acumulados."""
        labels = array(self.all_labels)
        preds = array(self.all_preds)
        probs = array(self.all_probs)

        acc = mean(labels == preds)
        mcc = matthews_corrcoef(labels, preds)
        kappa = cohen_kappa_score(labels, preds, weights="linear")
        matrix = confusion_matrix(labels, preds)

        metrics = {"acc": acc, "mcc": mcc, "kappa": kappa, "confusion_matrix": matrix}

        if self.num_classes == 2:
            metrics["roc_auc"] = roc_auc_score(labels, probs)
            metrics["f1"] = f1_score(labels, preds)

        return metrics

    def print_report(self, metrics):
        """Imprime o relatório formatado no console."""
        info("### eval report ###")
        info(f"ACC: {metrics['acc']:.4f}")
        info(f"MCC: {metrics['mcc']:.4f}")
        info(f"Cohen Kappa Score: {metrics['kappa']:.4f}")

        if self.num_classes == 2:
            info(f"ROC Score: {metrics.get('roc_auc', 0.0):.4f}")
            info(f"F1-Score: {metrics.get('f1', 0.0):.4f}")

        info(metrics["confusion_matrix"])
        info("###################")

    def plot_final_reports(self, roc_values, acc_values):
        """Gera os plots finais (Curva ROC e Matriz de Confusão Normalizada)."""
        labels = array(self.all_labels)
        preds = array(self.all_preds)
        probs = array(self.all_probs)
        matrix = confusion_matrix(labels, preds)

        if self.num_classes == 2:
            figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = roc_auc_score(labels, probs)
            plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
            xlim([0, 1])
            ylim([0, 1])
            xlabel("False Positive Rate")
            ylabel("True Positive Rate")
            title("ROC Curve")
            legend()
            show()

        cm_norm = matrix.astype("float") / matrix.sum(axis=1)[:, newaxis]
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_norm, display_labels=range(self.num_classes)
        )
        disp.plot()
        show()
