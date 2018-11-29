# Introduction

## Partition
```sh
$ sinfo
PARTITION        AVAIL  TIMELIMIT  NODES  STATE NODELIST
single_k40_node     up 1-00:00:00      2  down* tesla[11,17]
single_k40_node     up 1-00:00:00      9  alloc tesla[01-08,10]
single_k40_node     up 1-00:00:00      1   idle tesla09
single_k40_node     up 1-00:00:00      5   down tesla[12-16]
dual_v100_node      up 1-00:00:00     10  alloc tesla[18-23,25-28]
dual_v100_node      up 1-00:00:00      1   idle tesla24
single_v100_node    up 1-00:00:00      2   idle tesla[29-30]
skl_node            up 1-00:00:00      4  alloc skl[04-05,08-09]
skl_node            up 1-00:00:00      5   idle skl[01-03,06-07]
bigmem_node         up 1-00:00:00      2   idle bigmem,bigmem2
```
```sh
$ sinfo -Nel
Thu Nov 29 09:58:25 2018
NODELIST   NODES        PARTITION       STATE CPUS    S:C:T MEMORY TMP_DISK WEIGHT AVAIL_FE REASON
bigmem         1      bigmem_node        idle   40   4:10:1 516860        0      1   (null) none
bigmem2        1      bigmem_node        idle   56   4:14:1 775143        0      1   (null) none
skl01          1         skl_node        idle   36   2:18:1 192000        0      1 Xeon6140 none
skl02          1         skl_node        idle   36   2:18:1 192000        0      1 Xeon6140 none
skl03          1         skl_node        idle   36   2:18:1 192000        0      1 Xeon6140 none
skl04          1         skl_node   allocated   36   2:18:1 192000        0      1 Xeon6140 none
skl05          1         skl_node   allocated   36   2:18:1 192000        0      1 Xeon6140 none
skl06          1         skl_node        idle   36   2:18:1 192000        0      1 Xeon6140 none
skl07          1         skl_node        idle   36   2:18:1 192000        0      1 Xeon6140 none
skl08          1         skl_node   allocated   36   2:18:1 192000        0      1 Xeon6140 none
skl09          1         skl_node   allocated   36   2:18:1 192000        0      1 Xeon6140 none
tesla01        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla02        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla03        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla04        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla05        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla06        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla07        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla08        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla09        1  single_k40_node        idle   20   2:10:1 129031        0      1 TeslaK40 none
tesla10        1  single_k40_node   allocated   20   2:10:1 129031        0      1 TeslaK40 none
tesla11        1  single_k40_node       down*   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla12        1  single_k40_node        down   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla13        1  single_k40_node        down   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla14        1  single_k40_node        down   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla15        1  single_k40_node        down   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla16        1  single_k40_node        down   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla17        1  single_k40_node       down*   20   2:10:1 129031        0      1 TeslaK40 not_available
tesla18        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla19        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla20        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla21        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla22        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla23        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla24        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla25        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla26        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla27        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla28        1   dual_v100_node   allocated   20   2:10:1 129031        0      1 TeslaV10 none
tesla29        1 single_v100_node        idle   20   2:10:1 100000        0      1 TeslaV10 none
tesla30        1 single_v100_node        idle   20   2:10:1 100000        0      1 TeslaV10 none
```
