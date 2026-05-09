query_rewrite_cache = {}
answer_cache = {}


def get_cached_rewrite(question):
    return query_rewrite_cache.get(question)


def set_cached_rewrite(question, rewritten_query):
    query_rewrite_cache[question] = rewritten_query


def get_cached_answer(rewritten_query):
    return answer_cache.get(rewritten_query)


def set_cached_answer(rewritten_query, answer_data):
    answer_cache[rewritten_query] = answer_data
