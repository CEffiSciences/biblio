from os import environ as env, path, makedirs

if not path.exists(env.get("DATA_PATH")):
    makedirs(env.get("DATA_PATH"))