# !/usr/bin/python
# -*- coding: utf-8 -*-

# @Time: 2020/1/17 20:14
# @Author: Casually
# @File: json_utils.py
# @Email: fjkl@vip.qq.com
# @Software: PyCharm
#
# 功能：关于flask的jsonify序列化dict后的结果会自动根据key的下标升序排列,
#       希望dict原样返回的方法 [重写jsonify方法]

from flask import current_app, json


def jsonify(*args, **kwargs):
    """This function wraps :func:`dumps` to add a few enhancements that make
    life easier.  It turns the JSON output into a :class:`~flask.Response`
    object with the :mimetype:`application/json` mimetype.  For convenience, it
    also converts multiple arguments into an array or multiple keyword arguments
    into a dict.  This means that both ``jsonify(1,2,3)`` and
    ``jsonify([1,2,3])`` serialize to ``[1,2,3]``.

    For clarity, the JSON serialization behavior has the following differences
    from :func:`dumps`:

    1. Single argument: Passed straight through to :func:`dumps`.
    2. Multiple arguments: Converted to an array before being passed to
       :func:`dumps`.
    3. Multiple keyword arguments: Converted to a dict before being passed to
       :func:`dumps`.
    4. Both args and kwargs: Behavior undefined and will throw an exception.

    Example usage::

        from flask import jsonify

        @app.route('/_get_current_user')
        def get_current_user():
            return jsonify(username=g.user.username,
                           email=g.user.email,
                           id=g.user.id)

    This will send a JSON response like this to the browser::

        {
            "username": "admin",
            "email": "admin@localhost",
            "id": 42
        }


    .. versionchanged:: 0.11
       Added support for serializing top-level arrays. This introduces a
       security risk in ancient browsers. See :ref:`json-security` for details.

    This function's response will be pretty printed if the
    ``JSONIFY_PRETTYPRINT_REGULAR`` config parameter is set to True or the
    Flask app is running in debug mode. Compressed (not pretty) formatting
    currently means no indents and no spaces after separators.

    .. versionadded:: 0.2
    """

    indent = None
    separators = (',', ':')

    if current_app.config['JSONIFY_PRETTYPRINT_REGULAR'] or current_app.debug:
        indent = 2
        separators = (', ', ': ')

    if args and kwargs:
        raise TypeError('jsonify() behavior undefined when passed both args and kwargs')
    elif len(args) == 1:  # single args are passed directly to dumps()
        data = args[0]
    else:
        data = args or kwargs

    return current_app.response_class(
        json.dumps(data, indent=indent, separators=separators, sort_keys=False) + '\n',
        mimetype=current_app.config['JSONIFY_MIMETYPE']
    )
