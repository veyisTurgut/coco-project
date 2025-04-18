* run neo4j on server
    - `docker run --name neo4j-server -d -p 7474:7474 -p 7687:7687 -v $HOME/neo4j/data:/data -v $HOME/neo4j/logs:/logs -v $HOME/neo4j/conf:/var/lib/neo4j/conf -e NEO4J_AUTH=neo4j/${NEO4J_PASSWORD} neo4j:latest`
* neo4j url
    - `gcloud.madlen.io:7474`

# steps
- first generate data by running `generate_data.py`
- then load it to neo4j by running `load_graph.py`
- then analyze the data by running `analyze_graph.py`
