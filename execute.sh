echo "----------  ridge ----------"
python3 main.py --dataset="h36m" --path_to_data=. --method_name="ridge_regression" --use_cross_validation
echo "---------- logistic --------"
python3 main.py --dataset="h36m" --path_to_data=. --method_name="logistic_regression" --use_cross_validation