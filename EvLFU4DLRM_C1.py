import pandas as pd
import numpy as np
import time
import os

# EvLFU part
# Change here for higher precision!
EVPATH = "D:\\github\\ev-table-dlrm\\weights_and_biases\\epoch-0\\ev-table-4\\"
EvTable_C1 = dict()


def loadEvTable(ev_path):
    print("****************Loading embedding layers")

    print("****************Load new set of EV Table from = " + EVPATH)

    for ev_idx in range(0, 26):
        # Reference: Daniar's Github

        # Read new EV Table from file
        new_ev_path = os.path.join(ev_path,
                                   "ev-table-" + str(ev_idx + 1) + ".csv")
        new_ev_df = pd.read_csv(new_ev_path, dtype=float, delimiter=',')
        # Convert to numpy first before to tensor
        new_ev_arr = new_ev_df.to_numpy()
        # Convert to tensor
        print("*********************Loading NEW EV per embedding layer = " + new_ev_path)

        global EvTable_C1
        EvTable_C1[ev_idx + 1] = new_ev_arr

    print("****************All EvTable loaded in the Memory!")


def copycatCacheReplacementPolicy(key):
    tableNum, idx = key.split('-', 2)
    global EvTable_C1
    return EvTable_C1[int(tableNum)][int(idx), :]


###########################################EvLFU##########################################
cap_C1 = 500
min_C1 = 0
vals_C1 = dict()
counts_C1 = dict()
lists_C1 = dict()
lists_C1[0] = []
# flushing part:
nPerfectItem_C1 = 0
flushRate_C1 = 0.4
perfectItemCapacity_C1 = 1.0


def request(group_keys):
    aggHitMissRecord = []
    aggHit = 0
    global cap_C1, min_C1, vals_C1, counts_C1, lists_C1, nPerfectItem_C1, flushRate_C1, perfectItemCapacity_C1
    for key in group_keys:
        if vals_C1.get(key) is not None:
            aggHitMissRecord.append(True)
            aggHit += 1
        else:
            aggHitMissRecord.append(False)
    emb_weights = []
    for key in group_keys:
        val = update(key, aggHit, len(aggHitMissRecord))
        emb_weights.append(val)

    if lists_C1.get(26) and not len(lists_C1.get(26)) == 0:
        nPerfectItem_C1 = len(lists_C1.get(26))

    return aggHitMissRecord, emb_weights


def update(key, aggHit, nGroup):
    val = getValFromMem(key, aggHit)
    if val is None:
        val = getValFromStore(key)
        set(key, val, aggHit)
    return val


def getValFromMem(key, aggHit):  # Get From Mem
    global cap_C1, min_C1, vals_C1, counts_C1, lists_C1, nPerfectItem_C1, flushRate_C1, perfectItemCapacity_C1
    if vals_C1.get(key) is None:
        return None
    count = counts_C1.get(key)
    newCount = count
    if count < aggHit:
        newCount = aggHit
    counts_C1[key] = newCount
    lists_C1.get(count).remove(key)

    if count == min_C1:
        while (lists_C1.get(min_C1) is None) or len(lists_C1.get(min_C1)) == 0:
            min_C1 += 1
    if lists_C1.get(newCount) is None:
        lists_C1[newCount] = []
        lists_C1 = dict(sorted(lists_C1.items()))
    lists_C1.get(newCount).append(key)
    return vals_C1[key]


def getValFromStore(key):
    # key Format:
    # #EVTABLE-#INDEX
    val = copycatCacheReplacementPolicy(key)
    return val


def set(key, value, aggHit):
    global cap_C1, min_C1, vals_C1, counts_C1, lists_C1, nPerfectItem_C1, flushRate_C1, perfectItemCapacity_C1
    if cap_C1 <= 0:
        return
    if vals_C1.get(key) is not None:
        vals_C1[key] = value
        getValFromMem(key, aggHit)
        return

    # Flushing:
    if nPerfectItem_C1 >= int(cap_C1 * perfectItemCapacity_C1):
        # print("flushing!")
        for i in range(0, int(flushRate_C1 * cap_C1) + 1):
            evictKey = lists_C1.get(26)[0]
            lists_C1.get(26).remove(evictKey)
            vals_C1.pop(evictKey)
            counts_C1.pop(evictKey)

        nPerfectItem_C1 = len(lists_C1.get(26))
        if len(vals_C1) < cap_C1:
            min_C1 = aggHit

    # key allows to insert in the cache:
    if aggHit >= min_C1:
        if len(vals_C1) >= cap_C1:
            evictKey = lists_C1.get(min_C1)[0]
            lists_C1.get(min_C1).remove(evictKey)
            vals_C1.pop(evictKey)
            counts_C1.pop(evictKey)
        # If the key is new, insert the value:
        vals_C1[key] = value
        counts_C1[key] = aggHit

        if lists_C1.get(aggHit) is None:
            lists_C1[aggHit] = []
            lists_C1 = dict(sorted(lists_C1.items()))
        lists_C1.get(aggHit).append(key)
    else:
        min_C1 = aggHit
        if len(vals_C1) < cap_C1:
            vals_C1[key] = value
            counts_C1[key] = aggHit

            if lists_C1.get(aggHit) is None:
                lists_C1[aggHit] = []
                lists_C1 = dict(sorted(lists_C1.items()))
            lists_C1.get(aggHit).append(key)
    while (lists_C1.get(min_C1) is None) or len(lists_C1.get(min_C1)) == 0:
        min_C1 += 1


def main():
    global cap_C1, min_C1, vals_C1, counts_C1, lists_C1, nPerfectItem_C1, flushRate_C1, perfectItemCapacity_C1
    # giving workload:
    workload_dir = "D:\\github\\EV\\cache-benchmark\\Archive-new-0.5M\\"
    workload_files = []
    workload_files.append("workload-group-1.csv")
    workload_files.append("workload-group-2.csv")
    workload_files.append("workload-group-3.csv")
    workload_files.append("workload-group-5.csv")
    workload_files.append("workload-group-10.csv")
    workload_files.append("workload-group-11.csv")
    workload_files.append("workload-group-12.csv")
    workload_files.append("workload-group-20.csv")
    workload_files.append("workload-group-21.csv")
    workload_files.append("workload-group-22.csv")
    workload_files.append("workload-group-23.csv")

    workload_files.append("workload-group-4.csv")
    workload_files.append("workload-group-6.csv")
    workload_files.append("workload-group-7.csv")
    workload_files.append("workload-group-8.csv")
    workload_files.append("workload-group-9.csv")
    workload_files.append("workload-group-13.csv")
    workload_files.append("workload-group-14.csv")
    workload_files.append("workload-group-15.csv")
    workload_files.append("workload-group-16.csv")
    workload_files.append("workload-group-17.csv")
    workload_files.append("workload-group-18.csv")
    workload_files.append("workload-group-19.csv")
    workload_files.append("workload-group-24.csv")
    workload_files.append("workload-group-25.csv")
    workload_files.append("workload-group-26.csv")

    nTableWorkload = len(workload_files)
    arrRawWorkload = []

    # read all workloads:
    for workload_file in workload_files:
        workload = np.asarray(pd.read_csv(workload_dir + workload_file, header=None).values[0:500000, 0])
        arrRawWorkload.append(workload)

    arrRawWorkload = np.asarray(arrRawWorkload)
    # print(arrRawWorkload.shape)
    # merge the workloads
    arrMergedWorkload = np.stack(arrRawWorkload, axis=1)
    groupedWorkloadKeys = arrMergedWorkload
    print(arrMergedWorkload.shape)
    print("Done merging ALL workloads: total = ", arrRawWorkload.shape[0], 'rows')

    # Run the Alg:

    start_time = time.time()
    prefectHit = 0
    countR = 0
    for groupKeys in groupedWorkloadKeys:
        aggHitMissRecord, _ = request(groupKeys)
        flag = True
        for isHit in aggHitMissRecord:
            if not isHit:
                flag = False
                break
        if flag:
            prefectHit += 1
    print(prefectHit)
    print("time:")
    print(time.time() - start_time)


if __name__ == '__main__':
    loadEvTable(EVPATH)
    main()
