# NewsClassification
News classification for Codenrock

1. Run comand "docker build -t news_class ./Docker"
2. Run comand "docker run -d --name mycontainer -p 80:80 myimage"
3. Open link http://localhost:8008/ to test
4. Open link http://192.168.99.100:8008/docs or http://127.0.0.1:8008/docs (or equivalent, using your Docker host)
4. Open link http://192.168.99.100:8008/redoc or http://127.0.0.1:8008/redoc (or equivalent, using your Docker host)
5. Upload your csv file via upload panel
6. Check status of calculation in root panel
7. When status is "Completed!" - Download result via download panel
