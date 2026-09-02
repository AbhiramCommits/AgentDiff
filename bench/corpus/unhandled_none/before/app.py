def find_user(users, name):
    for user in users:
        if user["name"] == name:
            return user
    return None


def get_email(users, name):
    user = find_user(users, name)
    if user is None:
        return None
    return user["email"]
