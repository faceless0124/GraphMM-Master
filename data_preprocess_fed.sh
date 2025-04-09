cd data_preprocess_fed

city='beijing'

python build_road_graph.py $city
echo 'finish build_road_graph'

python build_A.py $city
echo 'finish build_A'

city='beijing'
client_num=15
sample_rates="0.1 0.1 0.1 0.2 0.2 0.2 0.3 0.3 0.3 0.4 0.4 0.4 0.5 0.5 0.5"
input_path='input_15'
output_path='output_15'

python data_process.py $city $client_num "$sample_rates" $input_path $output_path
echo 'finish data_process'

python build_trace_graph.py $city $output_path
echo 'finish build_trace_graph'

python build_trace_graph_client.py $city $output_path
echo 'finish build_trace_graph_client'

python maproad2grid.py $city $output_path
echo 'finish maproad2grid'

python maproad2grid_client.py $city $output_path
echo 'finish maproad2grid_client'

python build_grid_road_matrix.py $city $output_path
echo 'finish build_grid_road_matrix'