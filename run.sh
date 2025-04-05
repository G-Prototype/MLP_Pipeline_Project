#!/bin/bash

specs_output=$(python3 -u Specifications.py | tee /dev/tty)

echo -e "\nRunning parallel training..."
output_parallel=$(torchrun --nproc_per_node 2 ManualPipe.py | tee /dev/tty)
last_line_parallel=$(echo "$output_parallel" | tail -n 1)
training_time_parallel=$(echo "$last_line_parallel" | grep -o '[0-9.]\+' | awk '{printf "%.2f", $1}')

echo -e "\nRunning regular training..."
output_regular=$(torchrun Regular.py | tee /dev/tty)
last_line_regular=$(echo "$output_regular" | tail -n 1)
training_time_regular=$(echo "$last_line_regular" | grep -o '[0-9.]\+' | awk '{printf "%.2f", $1}')



speedup=0
improvement=0

speedup=$(echo "scale=2; $training_time_regular / $training_time_parallel" | bc -l)
improvement_percentage=$(echo "scale=2; (($training_time_regular - $training_time_parallel) / $training_time_parallel) * 100" | bc -l)

echo -e "\n=== Performance Results ==="
echo "Regular Training Time: $training_time_regular seconds"
echo "Parallel Training Time: $training_time_parallel seconds"
echo "Speedup: ${speedup}x"
echo "Improvement: ${improvement_percentage}%"

echo -e "\nSaving results to System_Specifications_and_Results.txt"
{
    echo "=== System Specifications ==="
    echo "$specs_output"
    echo -e "\n=== Performance Results ==="
    echo "Regular Training Time: $training_time_regular seconds"
    echo "Parallel Training Time: $training_time_parallel seconds"
    echo "Speedup: ${speedup}x"
    echo "Improvement: ${improvement_percentage}%"
    echo "Date: $(date)"
} > System_Specifications_and_Results.txt