import faery
import neuromorphic_drivers as nd

configuration = nd.prophesee_evk4.Configuration(
    biases=nd.prophesee_evk4.Biases(diff_off=170, diff_on=130)
)

faery.events_stream_from_camera(
    driver="NeuromorphicDrivers",
    nd_configuration=configuration,
).to_file("recording.es")
