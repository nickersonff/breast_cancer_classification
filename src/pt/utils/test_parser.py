import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from pt.orchestrator import init_datalist_parser
from pt.utils.constants import Constants


# if __name__ == "__main__":
def tester():
    config = Constants.get_config()
    datalist = init_datalist_parser(config=config)

    # Faz o split com o stratified group k fold
    indices = np.arange(len(datalist))
    labels = np.array([item["label"] for item in datalist])
    # Extrai o ID do paciente de cada dicionário para formar os grupos
    groups = np.array([item["patient_id"] for item in datalist])

    # 3. Configura o GroupKFold (garante que o mesmo patient_id não se repita entre treino e validação)
    n_splits = 5
    gkf = StratifiedGroupKFold(n_splits=n_splits)

    # 4. Loop de Validação Cruzada
    # Note que passamos 'groups' no método .split()
    for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, labels, groups)):
        print(f"\n--- Início do Fold {fold + 1}/{n_splits} ---")
        # Cria as sub-listas de dicionários para este fold
        fold_train_data = [datalist[i] for i in train_idx]
        fold_val_data = [datalist[i] for i in val_idx]

        train_patient_ids = np.unique([item["patient_id"] for item in fold_train_data])
        val_patient_ids = np.unique([item["patient_id"] for item in fold_val_data])

        # Encontra os IDs que estão presentes em ambos os arrays
        common_patients = np.intersect1d(train_patient_ids, val_patient_ids)
        # Verifica se existe pelo menos um elemento em comum
        if len(common_patients) > 0:
            print(
                f"⚠️ Atenção! Existem {len(common_patients)} pacientes em comum entre treino e validação:"
            )
            print(common_patients)
        else:
            print(
                "✅ Tudo certo! Não há vazamento de pacientes (nenhum ID se repete entre treino e validação)."
            )
