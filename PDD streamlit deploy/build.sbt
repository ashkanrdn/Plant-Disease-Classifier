WebKeys.webTarget := target.value / "scala-web"

artifactPath in PlayKeys.playPackageAssets := WebKeys.webTarget.value / (artifactPath in PlayKeys.playPackageAssets).value.getName