import pandas as pd
import os


def write_to_csv(filename, vals):
    try:
        do_header = not os.path.isfile(filename)

        if do_header:
            try:
                os.makedirs(os.path.split(filename)[0])
            except:
                pass

        now = pd.datetime.now()
        df = pd.DataFrame(vals, index=pd.date_range(now, periods=1))

        with open(filename, 'a') as f:
            df.to_csv(f, header=do_header)

    except:
        print ("Could not write to file %s." % filename)
