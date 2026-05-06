# -- coding: utf-8 --
from models.inits import *


def seq2instance(
    data, P, Q, low_index=0, high_index=100, granularity=15, sites=108, type="train"
):
    """
    :param data:
    :param P:
    :param Q:
    :param low_index:
    :param high_index:
    :param granularity:
    :param sites:
    :param type:
    :return: (N, sites, P) (N, sites, P+Q) (N, sites, P+Q) (N, sites, P+Q) (N, sites, P+Q) (N, 207, 24) (N, sites, P+Q)
    """
    X, DoW, D, H, M, L, XAll = [], [], [], [], [], [], []
    total_week_len = 60 // granularity * 24 * 7

    # Pre-process date information for all timestamps
    unique_times = {}
    for i in range(data.shape[0]):
        t_idx = int(data[i, 0])
        if t_idx not in unique_times:
            date_str = data[i, 1]
            day = int(data[i, 2])
            hour = int(data[i, 3])
            minute = int(data[i, 4])

            # Parse date once per unique timestamp
            parts = date_str.split("/")
            year, month, day_of_month = int(parts[0]), int(parts[1]), int(parts[2])
            try:
                dt = datetime.date(year, month, day_of_month)
                dow = dt.weekday()
            except:
                dow = 0  # Default to Monday if parsing fails

            unique_times[t_idx] = {
                "dow": dow,
                "day": day,
                "hour": hour,
                "minute": minute,
            }

    while low_index + P + Q < high_index:
        label = data[low_index * sites : (low_index + P + Q) * sites, -1:]
        label = np.concatenate(
            [label[i * sites : (i + 1) * sites] for i in range(Q + P)], axis=1
        )

        # Get time info from pre-processed dictionary
        time_info = unique_times.get(
            low_index, {"dow": 0, "day": 1, "hour": 0, "minute": 0}
        )

        X.append(
            np.reshape(
                data[low_index * sites : (low_index + P) * sites, -1], [1, P, sites]
            )
        )

        # Create day of week array
        dow_arr = np.zeros((1, P + Q, sites))
        for j in range(P + Q):
            t = low_index + j
            if t in unique_times:
                dow_arr[0, j, :] = unique_times[t]["dow"]
        DoW.append(dow_arr)

        # Day of month
        D.append(np.ones((1, P + Q, sites)) * time_info["day"])

        # Hour
        H.append(np.ones((1, P + Q, sites)) * time_info["hour"])

        # Minute
        minutes_index_of_day = time_info["hour"] * 60 + time_info["minute"]
        M.append(np.ones((1, P + Q, sites)) * (minutes_index_of_day // granularity))

        L.append(np.reshape(label, [1, sites, Q + P]))

        # XAll - past data with weekly history
        start_idx = max(0, (low_index - total_week_len) * sites)
        end_idx = min(data.shape[0], (low_index - total_week_len + P + Q) * sites)
        if end_idx - start_idx == (P + Q) * sites:
            XAll.append(np.reshape(data[start_idx:end_idx, -1], [1, P + Q, sites]))
        else:
            # Pad if not enough historical data
            x_all = np.zeros((1, P + Q, sites))
            available = min(end_idx - start_idx, (P + Q) * sites)
            if available > 0:
                x_all_data = data[start_idx : start_idx + available, -1]
                x_all.flat[:available] = x_all_data.flatten()
            XAll.append(x_all)

        if type == "train":
            low_index += 1
        else:
            low_index += 1

    if len(X) == 0:
        # Return empty arrays if no data
        return (
            np.zeros((0, P, sites)),
            np.zeros((0, P + Q, sites)),
            np.zeros((0, P + Q, sites)),
            np.zeros((0, P + Q, sites)),
            np.zeros((0, P + Q, sites)),
            np.zeros((0, sites, P + Q)),
            np.zeros((0, P + Q, sites)),
        )

    return (
        np.concatenate(X, axis=0),
        np.concatenate(DoW, axis=0),
        np.concatenate(D, axis=0),
        np.concatenate(H, axis=0),
        np.concatenate(M, axis=0),
        np.concatenate(L, axis=0),
        np.concatenate(XAll, axis=0),
    )


def loadData(args):
    # Traffic
    df = pd.read_csv(args.file_train_s)
    Traffic = df.values
    # train/val/test
    total_samples = df.shape[0] // args.site_num

    train_low = 60 // args.granularity * 24 * 7
    val_low = round(args.train_ratio * total_samples)
    test_low = round((args.train_ratio + args.validate_ratio) * total_samples)

    # X, Y, day of week, day, hour, minute, label, all X
    trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll = seq2instance(
        Traffic,
        args.input_length,
        args.output_length,
        low_index=train_low,
        high_index=val_low,
        granularity=args.granularity,
        sites=args.site_num,
        type="train",
    )
    print("training dataset has been loaded!")
    valX, valDoW, valD, valH, valM, valL, valXAll = seq2instance(
        Traffic,
        args.input_length,
        args.output_length,
        low_index=val_low,
        high_index=test_low,
        granularity=args.granularity,
        sites=args.site_num,
        type="validation",
    )
    print("validation dataset has been loaded!")
    testX, testDoW, testD, testH, testM, testL, testXAll = seq2instance(
        Traffic,
        args.input_length,
        args.output_length,
        low_index=test_low,
        high_index=total_samples,
        granularity=args.granularity,
        sites=args.site_num,
        type="test",
    )
    print("testing dataset has been loaded!")
    # normalization
    min, max = np.mean(trainX), np.std(trainX)
    trainX, trainXAll = (trainX - min) / (max), (trainXAll - min) / (max)
    valX, valXAll = (valX - min) / (max), (valXAll - min) / (max)
    testX, testXAll = (testX - min) / (max), (testXAll - min) / (max)

    # Debug print (commented out to avoid index errors with small datasets)
    # if len(testD) > 0:
    #     print(testD[0,12,0], testH[0,12,0], testM[0,12,0])
    return (
        trainX,
        trainDoW,
        trainD,
        trainH,
        trainM,
        trainL,
        trainXAll,
        valX,
        valDoW,
        valD,
        valH,
        valM,
        valL,
        valXAll,
        testX,
        testDoW,
        testD,
        testH,
        testM,
        testL,
        testXAll,
        min,
        max,
    )


# trainX, trainDoW, trainD, trainH, trainM, trainL, trainXAll, valX, valDoW, valD, valH, valM, valL, valXAll, testX, testDoW, testD, testH, testM, testL, testXAll, mean, std = loadData(para)
#
# print(trainX.shape, valX.shape, testX.shape)
