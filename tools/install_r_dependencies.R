local_library <- file.path(getwd(), ".r-library")
dir.create(local_library, showWarnings = FALSE)

install.packages(
  c("gamlss.dist", "gamlss"),
  lib = local_library,
  repos = "https://cloud.r-project.org"
)
