stats
- size, int
- filetype, cat
- bucket, cat
- ppi check, bool
- source, cat
- creation date, time
- permission, int
- migrations []
  - timestamp, time
  - bucket, cat
- access []
  - rw, bool
  - timestamp, time
  - requestor, int
  - lat, time

API
- list
    inputs (filter_by_migrations=[requestor, timestamp])
    outputs ([key])
- get
    inputs (key, requestor)
    outputs (value)
- put
    inputs (key, value, requestor)
    outputs (success)
- stats
    inputs (key)
    outputs (stats structs)
# hackaithon2
