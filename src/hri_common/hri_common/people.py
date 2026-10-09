"""Shared registry of enrolled people (single source of truth for names)."""

import os
import re
import shutil
import time

import numpy as np
import yaml

FACE_FILE = 'face.npy'
VOICE_FILE = 'voice.npy'
DOC_FILE = 'doc.txt'
KEEP_BACKUPS = 10


def people_dir():
    """Returns the people directory (override with HRI_PEOPLE_DIR)."""
    return os.path.expanduser(os.environ.get('HRI_PEOPLE_DIR', '~/hri_people'))


def person_path(person_id, filename):
    """Returns the path of a per-person file (creates nothing)."""
    return os.path.join(people_dir(), person_id.lower(), filename)


def _read_yaml():
    path = os.path.join(people_dir(), 'people.yaml')
    if not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _write_yaml(data):
    os.makedirs(people_dir(), exist_ok=True)
    with open(os.path.join(people_dir(), 'people.yaml'), 'w',
              encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def load_people():
    """Returns {person_id: display_name} from people.yaml."""
    people = _read_yaml().get('people') or {}
    return {pid.lower(): (info or {}).get('display_name', pid.capitalize())
            for pid, info in people.items()}


def ensure_person(person_id, display_name=None):
    """Registers a person in people.yaml (if missing) and creates their folder."""
    person_id = person_id.lower()
    if person_id == 'unknown' or not re.fullmatch(r'[a-z0-9_]+', person_id):
        raise ValueError(
            "person_id: only lowercase letters, digits and _ (and not 'unknown')")
    os.makedirs(os.path.join(people_dir(), person_id), exist_ok=True)
    data = _read_yaml()
    people = data.get('people') or {}
    data['people'] = people
    if person_id not in people:
        people[person_id] = {
            'display_name': display_name or person_id.capitalize()}
        _write_yaml(data)
    return person_id


def set_display_name(person_id, display_name):
    """Updates the display name of a registered person."""
    person_id = person_id.lower()
    data = _read_yaml()
    people = data.get('people') or {}
    if person_id not in people:
        raise KeyError(f'{person_id} no está registrado')
    people[person_id] = {**(people[person_id] or {}),
                         'display_name': display_name}
    data['people'] = people
    _write_yaml(data)


def remove_person(person_id):
    """Removes a person from people.yaml and deletes their folder."""
    person_id = person_id.lower()
    data = _read_yaml()
    people = data.get('people') or {}
    if person_id not in people:
        raise KeyError(f'{person_id} no está registrado')
    del people[person_id]
    data['people'] = people
    _write_yaml(data)
    shutil.rmtree(os.path.join(people_dir(), person_id), ignore_errors=True)


def clear_person_files(person_id, filenames):
    """Deletes the given data files of a person; returns the ones removed."""
    removed = []
    for filename in filenames:
        path = person_path(person_id, filename)
        if os.path.isfile(path):
            os.remove(path)
            removed.append(filename)
    return removed


def backups_dir():
    """Returns the directory that holds backups (sibling of the people dir)."""
    return people_dir().rstrip('/') + '_backups'


def list_backups():
    """Returns backup names, oldest first."""
    base = backups_dir()
    return sorted(os.listdir(base)) if os.path.isdir(base) else []


def _prune_backups():
    for old in list_backups()[:-KEEP_BACKUPS]:
        shutil.rmtree(os.path.join(backups_dir(), old), ignore_errors=True)


def backup_people(prune=True):
    """Copies the whole people dir into a timestamped backup.

    Returns the backup name, or None if there is nothing to back up.
    """
    src = people_dir()
    if not os.path.isdir(src):
        return None
    stamp = time.strftime('%Y%m%d_%H%M%S')
    name, n = stamp, 1
    while os.path.exists(os.path.join(backups_dir(), name)):
        name = f'{stamp}_{n}'
        n += 1
    shutil.copytree(src, os.path.join(backups_dir(), name))
    if prune:
        _prune_backups()
    return name


def restore_backup(name):
    """Restores a backup (the current state is saved as a new backup first)."""
    src = os.path.join(backups_dir(), name)
    if not os.path.isdir(src):
        raise FileNotFoundError(f'No existe el respaldo {name}')
    backup_people(prune=False)
    shutil.rmtree(people_dir(), ignore_errors=True)
    shutil.copytree(src, people_dir())
    _prune_backups()


def load_embeddings(filename):
    """Returns {person_id: embedding} for registered people that have filename."""
    embeddings = {}
    for person_id in load_people():
        path = person_path(person_id, filename)
        if os.path.isfile(path):
            embeddings[person_id] = np.load(path)
    return embeddings


def best_match(embedding, references, threshold):
    """Returns (person_id, score); 'Unknown' if no reference reaches threshold."""
    best_id, best_score = 'Unknown', 0.0
    for person_id, ref in references.items():
        score = float(np.dot(embedding, ref) /
                      (np.linalg.norm(embedding) * np.linalg.norm(ref)))
        if score > best_score:
            best_id, best_score = person_id, score
    if best_score < threshold:
        return 'Unknown', best_score
    return best_id, best_score
