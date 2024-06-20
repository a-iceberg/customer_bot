# sudo rm -rf ./data/ssl_cert.pem
# sudo rm -rf ./data/ssl_pkey.pem
# sudo cp /etc/letsencrypt/live/YOUR-DOMAIN.COM/fullchain.pem ./data/ssl_cert.pem
# sudo cp /etc/letsencrypt/live/YOUR-DOMAIN.COM/privkey.pem ./data/ssl_pkey.pem
# sudo chmod -R 777 ./data/ssl_cert.pem
# sudo chmod -R 777 ./data/ssl_pkey.pem

# sudo docker compose up --build --force-recreate
sudo docker compose down -v
sudo docker compose up --build -d --remove-orphans --force-recreate