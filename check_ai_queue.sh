#!/bin/bash

# Script to check AI partition queue usage
# Usage: ./check_ai_queue.sh

echo "=========================================="
echo "  AI Partition Queue Status Summary"
echo "=========================================="
echo ""

# Get partition summary
echo "--- Partition Overview ---"
showpartitions | grep -E "^[[:space:]]*ai[[:space:]]"
echo ""

# Get node status summary
echo "--- Node Status ---"
sinfo -p ai -o "%.15P %.5a %.10l %.6D %.6t %.8z %.6m %.8d %.6w %.8f %.20G"
echo ""
echo "Node State Legend:"
echo "  allocated/alloc - Node fully occupied, no resources available"
echo "  mixed           - Node partially occupied, can accept new jobs"
echo "  idle            - Node completely idle, ready for new jobs"
echo "  drained/drained*- Node draining, not accepting new jobs (maintenance)"
echo "  down            - Node down (fault or maintenance)"
echo "  planned         - Node planned (being configured)"
echo ""

# Count jobs
RUNNING_JOBS=$(squeue -p ai -t RUNNING -h | wc -l)
PENDING_JOBS=$(squeue -p ai -t PENDING -h | wc -l)
TOTAL_JOBS=$(squeue -p ai -h | wc -l)

echo "--- Job Statistics ---"
echo "Running jobs:  $RUNNING_JOBS"
echo "Pending jobs:  $PENDING_JOBS"
echo "Total jobs:    $TOTAL_JOBS"
echo ""

# Show running jobs
if [ $RUNNING_JOBS -gt 0 ]; then
    echo "--- Running Jobs (showing first 30) ---"
    squeue -p ai -t RUNNING -o "%.18i %.9P %.12j %.8u %.2t %.10M %.6D %.20R" | head -31
    echo ""
fi

# Show user distribution for running jobs
if [ $RUNNING_JOBS -gt 0 ]; then
    echo "--- Running Jobs by User ---"
    squeue -p ai -t RUNNING -o "%.8u" | tail -n +2 | sort | uniq -c | sort -rn
    echo ""
fi

# Show pending jobs (top reasons)
if [ $PENDING_JOBS -gt 0 ]; then
    echo "--- Pending Jobs (showing first 20) ---"
    squeue -p ai -t PENDING -o "%.18i %.9P %.12j %.8u %.2t %.10M %.6D %.20R" | head -21
    echo ""
    
    echo "--- Pending Reasons ---"
    squeue -p ai -t PENDING -o "%.20R" | tail -n +2 | sort | uniq -c | sort -rn | head -10
    echo ""
fi

# Show node details
echo "--- Node Details ---"
sinfo -p ai -N -o "%.10N %.8T %.10P %.6D %.8z %.8m %.8d %.20G" | head -25
echo ""

# Show your own jobs if any
MY_JOBS=$(squeue -p ai -u $USER -h | wc -l)
if [ $MY_JOBS -gt 0 ]; then
    echo "--- Your Jobs in AI Queue ---"
    squeue -p ai -u $USER -o "%.18i %.9P %.12j %.8u %.2t %.10M %.6D %.20R"
    echo ""
fi

echo "=========================================="

