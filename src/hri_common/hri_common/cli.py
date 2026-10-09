"""Command line tool to manage enrolled people (see people.sh)."""

import argparse
import os
import shlex
import shutil
import subprocess
import sys

from hri_common.people import (
    DOC_FILE, FACE_FILE, VOICE_FILE, backup_people, clear_person_files,
    ensure_person, list_backups, load_people, person_path, remove_person,
    restore_backup, set_display_name)

CHROMA_PATH = '~/rag_db'
STRATEGIES = ('sentence', 'paragraph', 'fixed', 'llm')
FILES = {'face': FACE_FILE, 'voice': VOICE_FILE, 'doc': DOC_FILE}

SELFTEST_ID = 'zz_selftest'
SELFTEST_DOC = (
    'Zorbo Quintana vive en la ciudad de Quilmes.\n'
    'Su mascota es un axolote llamado Fideo.\n'
    'Su color favorito es el naranja neón.\n'
    'Trabaja como afinador de pianos.\n')
SELFTEST_QUESTION = '¿Cómo se llama la mascota de Zorbo?'
SELFTEST_SECRET = 'Fideo'


# ---------------------------------------------------------------- helpers

def run_module(module, *args):
    """Runs a python module from another package in a subprocess."""
    return subprocess.call([sys.executable, '-m', module, *args])


def ingest(target, extra=()):
    """Re-indexes one person (or 'all') in ChromaDB."""
    flags = ['--all'] if target == 'all' else ['--name', target]
    code = run_module('rag_service.ingest_documents', *flags, *extra)
    if code == 0 and '--dry-run' not in extra:
        print('Listo. Reinicia rag_service_node para que vea los cambios.')
    return code


def drop_chunks(person_id):
    """Deletes a person's chunks from ChromaDB (warns instead of failing)."""
    try:
        import chromadb
        client = chromadb.PersistentClient(
            path=os.path.expanduser(CHROMA_PATH))
        client.get_collection('documents').delete(
            where={'person_id': person_id})
    except Exception as e:
        print(f'Aviso: no pude limpiar ChromaDB ({e})')


def safety_backup():
    """Saves a backup before any destructive operation."""
    name = backup_people()
    if name:
        print(f'Respaldo automático guardado: {name}')


def require_registered(person_id):
    """Exits with a message if the person is not in people.yaml."""
    if person_id.lower() not in load_people():
        sys.exit(f"'{person_id}' no está registrado. Usa: add {person_id}")
    return person_id.lower()


def has(person_id, filename):
    """Returns 'sí' if the person has that file, '-' otherwise."""
    return 'sí' if os.path.isfile(person_path(person_id, filename)) else '-'


def ask(prompt, default=None):
    """Asks for a text value; Enter keeps the default."""
    suffix = f' [{default}]' if default else ''
    return input(f'{prompt}{suffix}: ').strip() or default or ''


def ask_yes(prompt):
    """Asks a yes/no question (default: no)."""
    return input(f'{prompt} [s/N] ').strip().lower() in ('s', 'si', 'sí', 'y')


def clear_data(person_id, kinds, confirm=True):
    """Backs up, then deletes face/voice/doc data (keeps the registration)."""
    kinds = list(FILES) if 'all' in kinds else kinds
    if confirm and not ask_yes(
            f"¿Borrar {', '.join(kinds)} de {person_id}?"):
        print('Cancelado.')
        return 1
    safety_backup()
    removed = clear_person_files(person_id, [FILES[k] for k in kinds])
    if 'doc' in kinds:
        drop_chunks(person_id)
    print(f"Borrado: {', '.join(removed) or 'nada (no había datos)'}")
    print('Reinicia los nodos para que lo vean.')
    return 0


# --------------------------------------------------------------- commands

def cmd_list(args):
    """Lists registered people and which data each one has."""
    people = load_people()
    if not people:
        print('No hay personas registradas.')
        return 0
    print(f'{"id":<14}{"nombre":<22}{"cara":<6}{"voz":<6}{"doc":<6}')
    for pid, display in people.items():
        print(f'{pid:<14}{display:<22}{has(pid, FACE_FILE):<6}'
              f'{has(pid, VOICE_FILE):<6}{has(pid, DOC_FILE):<6}')
    return 0


def cmd_add(args):
    """Registers a person, optionally with a document."""
    existed = args.name.lower() in load_people()
    person_id = ensure_person(args.name, args.display)
    if existed and args.display:
        set_display_name(person_id, args.display)
    print(f"{'Actualizado' if existed else 'Registrado'}: {person_id}")
    if args.doc:
        shutil.copy(os.path.expanduser(args.doc),
                    person_path(person_id, DOC_FILE))
        if not args.no_ingest:
            return ingest(person_id)
    return 0


def cmd_set_doc(args):
    """Replaces a person's document and re-indexes it."""
    person_id = require_registered(args.name)
    shutil.copy(os.path.expanduser(args.file),
                person_path(person_id, DOC_FILE))
    print(f'Documento actualizado: {person_id}')
    return 0 if args.no_ingest else ingest(person_id)


def cmd_edit_doc(args):
    """Opens the document in $EDITOR and re-indexes if it changed."""
    person_id = require_registered(args.name)
    path = person_path(person_id, DOC_FILE)
    before = os.path.getmtime(path) if os.path.isfile(path) else None
    subprocess.call(shlex.split(os.environ.get('EDITOR', 'nano')) + [path])
    after = os.path.getmtime(path) if os.path.isfile(path) else None
    if after is None or after == before:
        print('Sin cambios, no se reindexa.')
        return 0
    return ingest(person_id)


def cmd_set_display(args):
    """Changes the display name of a person."""
    set_display_name(require_registered(args.name), args.display)
    print('Nombre actualizado.')
    return 0


def cmd_ingest(args):
    """Re-indexes one person, or everyone if no name is given."""
    extra = []
    if args.strategy:
        extra += ['--strategy', args.strategy]
    if args.max_chars:
        extra += ['--max-chars', str(args.max_chars)]
    if args.dry_run:
        extra.append('--dry-run')
    target = require_registered(args.name) if args.name else 'all'
    return ingest(target, extra)


def cmd_remove(args):
    """Deletes a person: folder, people.yaml entry and ChromaDB chunks."""
    person_id = require_registered(args.name)
    if not args.yes and not ask_yes(
            f'¿Borrar a {person_id} y todos sus datos?'):
        print('Cancelado.')
        return 1
    safety_backup()
    drop_chunks(person_id)
    remove_person(person_id)
    print(f'Eliminado: {person_id}')
    return 0


def cmd_clear(args):
    """Deletes selected data of a person, keeping their registration."""
    return clear_data(require_registered(args.name), args.kinds,
                      confirm=not args.yes)


def cmd_backup(args):
    """Creates a backup of all people now."""
    name = backup_people()
    print(f'Respaldo creado: {name}' if name else 'No hay nada que respaldar.')
    return 0


def cmd_backups(args):
    """Lists available backups."""
    names = list_backups()
    if not names:
        print('No hay respaldos.')
    for name in names:
        print(f'  {name}')
    return 0


def cmd_restore(args):
    """Restores a backup (default: the most recent) and re-indexes."""
    names = list_backups()
    name = args.name or (names[-1] if names else None)
    if not name:
        print('No hay respaldos.')
        return 1
    if not args.yes and not ask_yes(
            f'¿Restaurar {name}? (antes se guarda el estado actual)'):
        print('Cancelado.')
        return 1
    try:
        restore_backup(name)
    except FileNotFoundError as e:
        print(e)
        return 1
    print(f'Restaurado: {name}')
    return ingest('all')


def query_docs(model, col, person_id, question, n=3):
    """Returns the chunks retrieved for a question, filtered by person."""
    emb = model.encode('query: ' + question,
                       normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=[emb], n_results=n,
                    where={'person_id': person_id})
    return res['documents'][0] if res['documents'] else []


def cmd_selftest(args):
    """Checks per-person isolation of the RAG (no camera, mic or ROS needed)."""
    from sentence_transformers import SentenceTransformer
    import chromadb
    from rag_service.ingest_documents import DB_PATH, index_person

    others = [p for p in load_people() if p != SELFTEST_ID]
    ensure_person(SELFTEST_ID, 'Selftest')
    results, col = [], None
    try:
        with open(person_path(SELFTEST_ID, DOC_FILE), 'w',
                  encoding='utf-8') as f:
            f.write(SELFTEST_DOC)
        model = SentenceTransformer('intfloat/multilingual-e5-small')
        client = chromadb.PersistentClient(path=DB_PATH)
        col = client.get_or_create_collection(
            'documents', metadata={'hnsw:space': 'cosine'})
        index_person(SELFTEST_ID, model, col)

        own = ' '.join(query_docs(model, col, SELFTEST_ID, SELFTEST_QUESTION))
        results.append(('Recupera los datos de su propia persona',
                        SELFTEST_SECRET in own))
        for other in others:
            leaked = ' '.join(
                query_docs(model, col, other, SELFTEST_QUESTION))
            results.append((f'Sin fuga de datos hacia {other}',
                            SELFTEST_SECRET not in leaked))
    finally:
        drop_chunks(SELFTEST_ID)
        remove_person(SELFTEST_ID)
    if col is not None:
        left = col.get(where={'person_id': SELFTEST_ID})['ids']
        results.append(('Limpieza: sin restos de la prueba', left == []))

    print('\nAutoprueba del RAG:')
    for label, ok in results:
        print(f"  {'OK   ' if ok else 'FALLA'} {label}")
    passed = results and all(ok for _, ok in results)
    print('RESULTADO:', 'todo bien' if passed else 'HAY FALLAS')
    return 0 if passed else 1


def cmd_enroll_voice(args):
    """Enrolls a voice (needs microphone)."""
    extra = ['--display', args.display] if args.display else []
    return run_module('audio_pipeline.voice_embedding_generator',
                      '--name', args.name, *extra)


def cmd_enroll_face(args):
    """Enrolls a face (needs camera or --image)."""
    extra = ['--display', args.display] if args.display else []
    if args.image:
        extra += ['--image', args.image]
    return run_module('vision_pipeline.face_embedding_generator',
                      '--name', args.name, *extra)


# ------------------------------------------------------- interactive menu

def pick_person():
    """Shows a numbered list of people and returns the chosen id (or None)."""
    people = load_people()
    if not people:
        print('No hay personas registradas.')
        return None
    ids = list(people)
    for i, pid in enumerate(ids, 1):
        print(f'  {i}. {people[pid]} ({pid})')
    choice = input('Número (Enter para cancelar): ').strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(ids):
        return None
    return ids[int(choice) - 1]


def do_name(pid):
    """Interactive: change display name."""
    current = load_people()[pid]
    new = ask('Nuevo nombre para mostrar', current)
    if new == current:
        print('Sin cambios.')
        return
    set_display_name(pid, new)
    print('Nombre actualizado.')


def do_doc(pid):
    """Interactive: edit or import the document (re-indexes automatically)."""
    print('  1. Editar en el editor')
    print('  2. Importar desde un archivo .txt')
    choice = input('Opción: ').strip()
    if choice == '1':
        cmd_edit_doc(argparse.Namespace(name=pid))
    elif choice == '2':
        path = ask('Ruta del archivo')
        if not os.path.isfile(os.path.expanduser(path)):
            print('No encontré ese archivo.')
            return
        cmd_set_doc(argparse.Namespace(name=pid, file=path, no_ingest=False))
    else:
        print('Cancelado.')


def do_face(pid):
    """Interactive: enroll face from camera or from a photo."""
    print('  1. Con la cámara')
    print('  2. Desde una foto')
    choice = input('Opción: ').strip()
    image = None
    if choice == '2':
        image = os.path.expanduser(ask('Ruta de la foto'))
        if not os.path.isfile(image):
            print('No encontré esa foto.')
            return
    elif choice != '1':
        print('Cancelado.')
        return
    cmd_enroll_face(argparse.Namespace(name=pid, display=None, image=image))


def do_voice(pid):
    """Interactive: record 5 seconds and enroll the voice."""
    if ask_yes('Se grabarán 5 segundos; habla normal frente al micrófono. '
               '¿Listo?'):
        cmd_enroll_voice(argparse.Namespace(name=pid, display=None))


def do_clear(pid):
    """Interactive: delete face, voice, document or everything."""
    print('  1. Cara')
    print('  2. Voz')
    print('  3. Documento')
    print('  4. Todo (conserva el registro)')
    kinds = {'1': ['face'], '2': ['voice'], '3': ['doc'],
             '4': ['all']}.get(input('Opción: ').strip())
    if kinds:
        clear_data(pid, kinds)
    else:
        print('Cancelado.')


STEPS = [('nombre', do_name), ('documento', do_doc),
         ('cara', do_face), ('voz', do_voice)]


def menu_new():
    """Interactive: register a new person and optionally load their data."""
    raw = ask('ID (minúsculas, números o _; ej. luis)').lower()
    if not raw:
        return
    if raw in load_people():
        print(f"'{raw}' ya existe. Usa 'Actualizar usuario'.")
        return
    display = ask('Nombre para mostrar', raw.capitalize())
    try:
        pid = ensure_person(raw, display)
    except ValueError as e:
        print(f'ID inválido: {e}')
        return
    print(f'Registrado: {pid}')
    print('Carga sus datos ahora (puedes omitir cualquiera y hacerlo después):')
    for label, func in STEPS[1:]:
        if ask_yes(f'¿Cargar {label}?'):
            func(pid)


def menu_update():
    """Interactive: pick a person and update one or all of their data."""
    pid = pick_person()
    if not pid:
        return
    while True:
        print(f'\n--- Actualizar: {load_people()[pid]} ({pid}) ---')
        print('  a. Todo (pregunta paso por paso)')
        print('  b. Nombre')
        print('  c. Documento')
        print('  d. Cara')
        print('  e. Voz')
        print('  f. Borrar datos (cara / voz / documento)')
        print('  0. Volver')
        option = input('Opción: ').strip().lower()
        if option == 'a':
            for label, func in STEPS:
                if ask_yes(f'¿Actualizar {label}?'):
                    func(pid)
        elif option in ('b', 'c', 'd', 'e'):
            dict(zip('bcde', (f for _, f in STEPS)))[option](pid)
        elif option == 'f':
            do_clear(pid)
        elif option == '0':
            return
        else:
            print('Opción no válida.')


def menu_reindex():
    """Interactive: re-index everyone or one person with a chosen strategy."""
    print('  1. Todos')
    print('  2. Una persona')
    choice = input('Opción: ').strip()
    if choice == '1':
        target = 'all'
    elif choice == '2':
        target = pick_person()
        if not target:
            return
    else:
        return
    strategy = ask('Estrategia (sentence / paragraph / fixed / llm)', 'llm')
    if strategy not in STRATEGIES:
        print('Estrategia no válida.')
        return
    extra = ['--strategy', strategy]
    if ask_yes('¿Solo mostrar los chunks, sin guardar?'):
        extra.append('--dry-run')
    ingest(target, extra)


def menu_backups():
    """Interactive: create, list and restore backups."""
    while True:
        print('\n--- Respaldos ---')
        print('  1. Crear respaldo ahora')
        print('  2. Ver respaldos')
        print('  3. Restaurar uno')
        print('  0. Volver')
        option = input('Opción: ').strip()
        if option == '1':
            cmd_backup(None)
        elif option == '2':
            cmd_backups(None)
        elif option == '3':
            names = list_backups()
            if not names:
                print('No hay respaldos.')
                continue
            for i, name in enumerate(names, 1):
                print(f'  {i}. {name}')
            choice = input('Número (Enter = el más reciente, 0 = cancelar): ')
            choice = choice.strip()
            if choice == '0':
                continue
            if choice and not (choice.isdigit()
                               and 1 <= int(choice) <= len(names)):
                print('Opción no válida.')
                continue
            name = names[int(choice) - 1] if choice else None
            cmd_restore(argparse.Namespace(name=name, yes=False))
        elif option == '0':
            return
        else:
            print('Opción no válida.')


def interactive():
    """Main interactive menu."""
    try:
        while True:
            print('\n=== Gestión de personas ===')
            print('  1. Nuevo usuario')
            print('  2. Actualizar usuario')
            print('  3. Ver usuarios')
            print('  4. Eliminar usuario')
            print('  5. Reindexar documentos (elegir estrategia)')
            print('  6. Respaldos')
            print('  7. Autoprueba del RAG (aislamiento entre personas)')
            print('  0. Salir')
            option = input('Opción: ').strip()
            if option == '1':
                menu_new()
            elif option == '2':
                menu_update()
            elif option == '3':
                cmd_list(None)
            elif option == '4':
                pid = pick_person()
                if pid:
                    cmd_remove(argparse.Namespace(name=pid, yes=False))
            elif option == '5':
                menu_reindex()
            elif option == '6':
                menu_backups()
            elif option == '7':
                cmd_selftest(None)
            elif option == '0':
                break
            else:
                print('Opción no válida.')
    except (EOFError, KeyboardInterrupt):
        print()
    print('Si cambiaste algo, reinicia los nodos para que lo vean.')
    return 0


# ------------------------------------------------------------------- main

def main(argv=None):
    """Entry point of the people CLI."""
    parser = argparse.ArgumentParser(prog='people.sh', description=__doc__)
    sub = parser.add_subparsers(dest='command')

    p = sub.add_parser('list', help='lista personas y qué datos tienen')
    p.set_defaults(func=cmd_list)

    p = sub.add_parser('add', help='registra una persona (opcional: con doc)')
    p.add_argument('name')
    p.add_argument('--display')
    p.add_argument('--doc', help='archivo .txt con su documento')
    p.add_argument('--no-ingest', action='store_true')
    p.set_defaults(func=cmd_add)

    p = sub.add_parser('set-doc', help='reemplaza el documento de alguien')
    p.add_argument('name')
    p.add_argument('file')
    p.add_argument('--no-ingest', action='store_true')
    p.set_defaults(func=cmd_set_doc)

    p = sub.add_parser('edit-doc', help='abre el documento en $EDITOR')
    p.add_argument('name')
    p.set_defaults(func=cmd_edit_doc)

    p = sub.add_parser('set-display', help='cambia el nombre para mostrar')
    p.add_argument('name')
    p.add_argument('display')
    p.set_defaults(func=cmd_set_display)

    p = sub.add_parser('ingest', help='reindexa (sin nombre = todos)')
    p.add_argument('name', nargs='?')
    p.add_argument('--strategy', choices=STRATEGIES)
    p.add_argument('--max-chars', type=int)
    p.add_argument('--dry-run', action='store_true',
                   help='muestra los chunks sin guardar nada')
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser('remove', help='borra a una persona y sus datos')
    p.add_argument('name')
    p.add_argument('--yes', action='store_true')
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser(
        'clear', help='borra cara/voz/doc, conserva el registro')
    p.add_argument('name')
    p.add_argument('kinds', nargs='+', choices=['face', 'voice', 'doc', 'all'])
    p.add_argument('--yes', action='store_true')
    p.set_defaults(func=cmd_clear)

    p = sub.add_parser('backup', help='crea un respaldo ahora')
    p.set_defaults(func=cmd_backup)

    p = sub.add_parser('backups', help='lista los respaldos')
    p.set_defaults(func=cmd_backups)

    p = sub.add_parser(
        'restore', help='restaura un respaldo (default: último)')
    p.add_argument('name', nargs='?')
    p.add_argument('--yes', action='store_true')
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser('selftest', help='prueba el aislamiento del RAG')
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser('enroll-voice', help='graba y guarda la voz')
    p.add_argument('name')
    p.add_argument('--display')
    p.set_defaults(func=cmd_enroll_voice)

    p = sub.add_parser('enroll-face', help='guarda la cara (cámara o --image)')
    p.add_argument('name')
    p.add_argument('--display')
    p.add_argument('--image', help='foto en vez de cámara')
    p.set_defaults(func=cmd_enroll_face)

    args = parser.parse_args(argv)
    if not args.command:
        return interactive()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
