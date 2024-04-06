import sys
import cv2
from optimum.intel.openvino import OVStableDiffusionPipeline

def main(args):
    pipe = OVStableDiffusionPipeline.from_pretrained("OpenVINO/stable-diffusion-1-5-fp32", compile=False)
    pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
    pipe.to("GPU.0")
    pipe.compile()

    # check if prompt is passed as argument
    if len(args) < 2:
        print("Error: Please pass a prompt as command line argument.")
        return

    # use the argument passed from command line
    prompt = args[1]
    output = pipe(prompt, num_inference_steps=50, output_type="pil")

    # save the generated image
    output.images[0].save("result.png")

    # display the generated image using OpenCV
    img = cv2.imread("result.png")
    cv2.imshow("Generated Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
