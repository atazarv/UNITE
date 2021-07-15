## Introduction

EMA Module is a module for context-aware EMA triggering.

## Usage:

```Python
from pathlib import Path
import EMAModule

mod = EMAModule.MainModule()
datapath = '/Path/To/data_uniterctXXX-2020-11-09-08-41-01.csv'
user_id = 'uniterctXXX'

# Here is where Sample_{username}.csv, density_{username}.npy, bndrs_{username}.csv and userinfo_{username}.json files are stored.
filepath = Path('/Path/To/User/Metadata/Files/')

mod.main(datapath, filepath, user_id)
```
