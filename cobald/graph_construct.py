import os
import json
import networkx as nx


def load_cobald_annotation(filename):
    """Загружает аннотацию CoBaLD из JSON-файла."""
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None

    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Извлекаем саму аннотацию, которая хранится внутри ключа 'cobald_annotation'
    return data


def find_root_node(annotation):
    """Находит главную вершину (Root) графа по связи '0'."""
    # (Функция оставлена для справки, но не используется напрямую для построения всех вершин)
    for head_id, dependent_id, relation in annotation.get('deps_eud', []):
        if head_id == '0' and relation == 'root':
            return dependent_id
    return None


def build_cobald_nodes(annotation):
    G = nx.DiGraph()

    ids = annotation.get('ids', [])
    words = annotation.get('words', [])
    lemmas = annotation.get('lemmas', [])
    deepslots = annotation.get('deepslots', [])
    semclasses = annotation.get('semclasses', [])

    G.add_node('0', token='ROOT', lemma='<ROOT>', ds='<ROOT>', sc='<ROOT>')

    for i in range(len(words)):

        node_attributes = {
            "id": i,
            "cob_id": ids[i],
            "token": words[i],
            "lemma": lemmas[i],
            "ds": deepslots[i],  # Deep Slot
            "sc": semclasses[i]  # Semantic Class
        }

        G.add_node(ids[i], **node_attributes)

    return G


def build_cobald_edges(G, annotation):
    deps_eud = annotation.get('deps_eud', [])
    deepslots = annotation.get('deepslots', [])
    ids = annotation.get('ids', [])


    id_to_idx = {str(cob_id): i for i, cob_id in enumerate(ids)}

    for head_id, dependent_id, eud_rel in deps_eud:
        h_id = str(head_id)
        d_id = str(dependent_id)

        if d_id in id_to_idx:
            dependent_index = id_to_idx[d_id]
            ds_label = deepslots[dependent_index]

            edge_attributes = {
                "eud_rel": eud_rel,
                "ds_label": ds_label
            }

            G.add_edge(h_id, d_id, **edge_attributes)

    return G

