def build_query(user_id):
    return "SELECT * FROM users WHERE id = %s", (user_id,)
