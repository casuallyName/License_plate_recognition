# License plate recognition
## 接口
| POST | （http: //localhost:5000） |
|------|--------------------------|
| Body | formdata                 |
| Key  | image                    |

## 关于POST地址：

-------
本机测试：
```python
if __name__ == '__main__':
    app.run()
    # 默认值：
    # host=127.0.0.1（localhost）
    # port=5000
    # debug=false
```
-------
本地测试：
```python
if __name__ == '__main__':
    app.run(
        host='192.168.1.100' # 本地路由地址，局域网下的主机均可通过该地址完成POST请求
        port=5000
    )
```

-------
服务器部署：
```python
if __name__ == '__main__':
    from waitress import serve
    serve(app, host='  Server IP  ', port=5000)
```
