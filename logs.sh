#!/bin/sh

# Array of container names
containers="customer_bot_alex_test
customer_bot_vlad-test
customer_bot_cc_test
customer_bot"

# Display the list of container names
echo "Available containers:"
i=1
for container in $containers; do
  echo "$i. $container"
  i=$((i+1))
done

# Prompt the user to choose a container
echo -n "Enter the number of the container to view logs: "
read choice

# Validate the user's choice
if [ $choice -lt 1 ] || [ $choice -gt $(echo "$containers" | wc -l) ]; then
  echo "Invalid choice. Exiting."
  exit 1
fi

# Get the selected container name
selected_container=$(echo "$containers" | sed -n "${choice}p")

# View the logs of the selected container
echo "Viewing logs for container: $selected_container"
sudo docker logs -f "$selected_container"