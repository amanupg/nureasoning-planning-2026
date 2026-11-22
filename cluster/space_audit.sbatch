#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J jn_space
#SBATCH -N 1
#SBATCH --exclude=ins058,ins082,ins084,ins092,ins095
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -o /path/to/cluster/jepa-nureasoning/logs/space_audit_%j.out
# READ-ONLY space audit of THIS thread's folder only (other agents audit their own). Deletes nothing.
D=/path/to/cluster/jepa-nureasoning
df -h /insomnia001/depts/edu | tail -1
echo "== $D (depth 2)"; du -xh --max-depth=2 $D 2>/dev/null | sort -rh | head -30
echo "== files >= 100 MB"; find $D -xdev -type f -size +100M -printf '%s\t%TY-%Tm-%Td %TH:%TM\t%p\n' 2>/dev/null | sort -rn | awk -F'\t' '{printf "%7.2f GB  %s  %s\n", $1/1e9, $2, $3}'
echo AUDIT_DONE
